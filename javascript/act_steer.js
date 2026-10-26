// Activation Steering: word map clicks, right-click menu and auto refresh.
//
// The word map (lib_act_steer/wordmap.py) is plain HTML; a click on a word only switches its
// colour via CSS. Edits reach Python through a hidden textbox (#act_steer_<tab>_word): JS writes
// "action|word|value|time" into it and fires its input event (actions: select, set, reset, lock,
// removed, restored). Rules and layers from the menu (allow / prefer / place / place_rule) use
// #act_steer_<tab>_rule instead, a light event that only saves them, so the next one can follow. A click on the image looks up each word's lift at that spot in the grid
// embedded in .wm-box[data-grid] and highlights the words acting there. Removing a word edits the
// prompt textbox here in JS (one level of undo). After every job the server raises a counter
// (/actsteer/version); when it changes, or another gallery image is picked, the "show" button is
// clicked so the word map follows the gallery, and the hidden auto step button (auto optimize).

(function() {
    // a click on the image selects the strongest word there (the colour shown at that spot) and
    // lights up the words connected to it: at least 1.2x their image average and 75 % of the top one
    var MIN_LIFT = 1.05;  // below: no word stands out at this spot
    var HIT_LIFT = 1.2;
    var HIT_SHARE = 0.75;
    var MAX_HITS = 6;

    function tabOf(el) {
        return el.closest("#tab_img2img") ? "img2img" : "txt2img";
    }

    function send(tab, action, word, value) {
        var box = gradioApp().querySelector("#act_steer_" + tab + "_word textarea");
        if (!box) return;
        box.value = [action, word, value === undefined ? "" : value, Date.now()].join("|");
        updateInput(box);
    }

    // a rule from the menu goes to its own hidden textbox: saved instantly, the view follows later
    function sendRule(tab, action, word, other) {
        var box = gradioApp().querySelector("#act_steer_" + tab + "_rule textarea");
        if (!box) return;
        box.value = [action, word, other, Date.now()].join("|");
        updateInput(box);
    }

    // drop the pair from both words' data-rel (the re-rendered view will not list it either)
    function forgetRelation(boxEl, a, b) {
        [[a, b], [b, a]].forEach(function(p) {
            labelsOf(boxEl, p[0].toLowerCase()).forEach(function(l) {
                l.dataset.rel = (l.dataset.rel || "").split(",").filter(function(x) {
                    return x && x.slice(0, x.lastIndexOf(":")) !== p[1];
                }).join(",");
            });
        });
    }

    // ---- prompt editing ----------------------------------------------------------------

    var undoPrompt = {txt2img: null, img2img: null};

    function promptBox(tab) {
        return gradioApp().querySelector("#" + tab + "_prompt textarea");
    }

    function escapeRe(s) {
        return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }

    // the word (also when written as (word:x)) taken out of the prompt, outside <lora:...> style tags
    function removeWord(prompt, word) {
        var w = escapeRe(word);
        var weighted = new RegExp("\\(\\s*" + w + "\\s*:\\s*-?[\\d.]+\\s*\\)", "gi");
        var bare = new RegExp("(^|[^\\w])" + w + "(?![\\w])", "gi");
        var count = 0;
        var out = prompt.split(/(<[^>]*>)/).map(function(part) {
            if (part.charAt(0) === "<") return part;
            part = part.replace(weighted, function() { count++; return ""; });
            return part.replace(bare, function(m, pre) { count++; return pre; });
        }).join("");
        out = out.replace(/\(\s*\)/g, "").replace(/[ \t]+/g, " ").replace(/\s+,/g, ",")
            .replace(/,(\s*,)+/g, ",").replace(/\(\s+/g, "(").replace(/\s+\)/g, ")")
            .replace(/^[\s,]+|[\s,]+$/g, "");
        return {prompt: out, count: count};
    }

    function setPrompt(tab, text) {
        var box = promptBox(tab);
        if (!box) return;
        box.value = text;
        updateInput(box);
    }

    function removeFromPrompt(tab, word, root) {
        var box = promptBox(tab);
        if (!box) return;
        var res = removeWord(box.value, word);
        if (!res.count) {
            spotText(root, "'" + word + "' is not a separate word in the prompt (part of another word or " +
                     "added by an extension): nothing removed.");
            return;
        }
        undoPrompt[tab] = {prompt: box.value, word: word};
        setPrompt(tab, res.prompt);
        spotText(root, "Removed '" + word + "' from the prompt (right click: undo).");
        send(tab, "removed", word);
    }

    function undoRemove(tab) {
        var u = undoPrompt[tab];
        if (!u) return;
        undoPrompt[tab] = null;
        setPrompt(tab, u.prompt);
        send(tab, "restored", u.word);
    }

    function isLocked(tab, word) {
        var box = gradioApp().querySelector("#script_" + tab + "_activation_steering_locked textarea");
        if (!box) return false;
        return box.value.split(",").some(function(w) { return w.trim().toLowerCase() === word.toLowerCase(); });
    }

    // ---- clicks on the image -------------------------------------------------------------

    function gridOf(boxEl) {
        if (boxEl._wmGrid === undefined) {
            try {
                var g = JSON.parse(boxEl.dataset.grid || "null");
                if (g) {
                    var words = {};
                    Object.keys(g.words).forEach(function(k) {
                        var bin = atob(g.words[k]);
                        var a = new Uint8Array(bin.length);
                        for (var i = 0; i < bin.length; i++) a[i] = bin.charCodeAt(i);
                        words[k] = a;
                    });
                    g.words = words;
                }
                boxEl._wmGrid = g;
            } catch (e) {
                boxEl._wmGrid = null;
            }
        }
        return boxEl._wmGrid;
    }

    // [[key, lift]] of the words acting at a spot (fx, fy = 0..1 of the image), strongest first;
    // the first is the strongest word there, the rest are connected to the spot
    function wordsAt(boxEl, fx, fy) {
        var g = gridOf(boxEl);
        if (!g) return [];
        var gx = Math.min(g.w - 1, Math.max(0, Math.floor(fx * g.w)));
        var gy = Math.min(g.h - 1, Math.max(0, Math.floor(fy * g.h)));
        var all = [];
        Object.keys(g.words).forEach(function(k) {
            all.push([k, g.words[k][gy * g.w + gx] * 4 / 255]);
        });
        all.sort(function(a, b) { return b[1] - a[1]; });
        if (!all.length || all[0][1] < MIN_LIFT) return [];
        var top = all[0][1];
        return [all[0]].concat(all.slice(1).filter(function(h) {
            return h[1] >= HIT_LIFT && h[1] >= top * HIT_SHARE;
        })).slice(0, MAX_HITS);
    }

    function spotOf(img, e) {
        var r = img.getBoundingClientRect();
        return [(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height];
    }

    function labelsOf(boxEl, key) {
        return Array.prototype.filter.call(boxEl.querySelectorAll(".wm-w[data-key]"), function(l) {
            return l.dataset.key === key;
        });
    }

    function wordOf(boxEl, key) {
        var l = labelsOf(boxEl, key)[0];
        return l ? l.dataset.word : key;
    }

    function spotText(root, text) {
        var s = root && root.querySelector(".wm-spot");
        if (s) s.textContent = text;
    }

    function clearSpot(boxEl) {
        boxEl.classList.remove("wm-spotted");
        boxEl.querySelectorAll(".wm-hit").forEach(function(l) { l.classList.remove("wm-hit"); });
        var dot = boxEl.querySelector(".wm-dot");
        if (dot) dot.style.display = "none";
    }

    var lastSpot = {};  // tab -> {fx, fy, box, time}: re-applied when the view is rendered again

    // mark the spot, light up its words; select = also select the strongest word (like its label)
    function showSpot(img, fx, fy, select) {
        var boxEl = img.closest(".wm-box");
        var tab = tabOf(boxEl);
        clearSpot(boxEl);
        var hits = wordsAt(boxEl, fx, fy);
        var dot = boxEl.querySelector(".wm-dot");
        if (dot) {
            dot.style.left = (img.offsetLeft + fx * img.offsetWidth) + "px";
            dot.style.top = (img.offsetTop + fy * img.offsetHeight) + "px";
            dot.style.display = "block";
        }
        lastSpot[tab] = {fx: fx, fy: fy, box: boxEl, time: Date.now()};
        if (!hits.length) {
            spotText(boxEl, "No word stands out at this spot.");
            return hits;
        }
        var top = hits[0][1];
        boxEl.classList.add("wm-spotted");
        hits.forEach(function(h) {
            labelsOf(boxEl, h[0]).forEach(function(l) {
                l.classList.add("wm-hit");
                l.style.setProperty("--wm-h", (h[1] / top).toFixed(2));
            });
        });
        spotText(boxEl, "At this spot: " + hits.map(function(h, i) {
            return (i ? "" : "▶ ") + wordOf(boxEl, h[0]) + " " + h[1].toFixed(1) + "x";
        }).join(", ") + "  (attention here vs. the word's average; ▶ selected)");
        if (select) {
            var label = labelsOf(boxEl, hits[0][0])[0];
            if (label) label.click();  // shows only its colour and selects it for the influence slider
        }
        return hits;
    }

    // the view was rendered again (after a selection): put the spot back on the new one
    onAfterUiUpdate(function() {
        Object.keys(lastSpot).forEach(function(tab) {
            var sp = lastSpot[tab];
            var boxEl = gradioApp().querySelector("#tab_" + tab + " .wm-box");
            if (!boxEl || boxEl === sp.box || Date.now() - sp.time > 15000) return;
            var img = boxEl.querySelector("img.wm-base");
            if (img && img.complete && img.offsetWidth) showSpot(img, sp.fx, sp.fy, false);
        });
    });

    // ---- right-click menu ----------------------------------------------------------------

    var menu = null;

    function closeMenu() {
        if (menu) menu.remove();
        menu = null;
    }

    function menuStyle() {
        if (document.getElementById("wm-menu-style")) return;
        var st = document.createElement("style");
        st.id = "wm-menu-style";
        st.textContent =
            ".wm-menu{position:fixed;z-index:10000;min-width:230px;max-width:360px;padding:4px 0;font-size:14px;" +
            // taller than the window (an area with several words): its own scrollbar, the page stays put
            "max-height:calc(100vh - 8px);box-sizing:border-box;overflow-y:auto;overscroll-behavior:contain;" +
            "background:var(--background-fill-primary,#222);color:var(--body-text-color,#eee);" +
            "border:1px solid var(--border-color-primary,#555);border-radius:8px;box-shadow:0 6px 24px rgba(0,0,0,.45)}" +
            ".wm-menu .wm-mh{padding:6px 12px 4px;font-weight:bold;display:flex;align-items:center;gap:6px}" +
            ".wm-menu .wm-mh i{display:inline-block;width:12px;height:12px;border-radius:3px}" +
            ".wm-menu .wm-mh small{font-weight:normal;opacity:.7}" +
            ".wm-menu .wm-mi{padding:5px 12px 5px 30px;cursor:pointer;white-space:nowrap}" +
            ".wm-menu .wm-mi:hover{background:var(--button-secondary-background-fill-hover,#444)}" +
            ".wm-menu .wm-mrow{display:flex;flex-wrap:wrap;gap:4px;padding:2px 12px 6px 30px}" +
            ".wm-menu .wm-mrow span{padding:2px 7px;border:1px solid var(--border-color-primary,#555);" +
            "border-radius:5px;cursor:pointer}" +
            ".wm-menu .wm-mrow span:hover{background:var(--button-secondary-background-fill-hover,#444)}" +
            ".wm-menu hr{margin:4px 0;border:0;border-top:1px solid var(--border-color-primary,#555)}";
        document.head.appendChild(st);
    }

    function item(parent, text, fn, chip) {
        var d = document.createElement(chip ? "span" : "div");
        if (!chip) d.className = "wm-mi";
        d.textContent = text;
        d.addEventListener("click", function(ev) {
            ev.stopPropagation();
            closeMenu();
            fn();
        });
        parent.appendChild(d);
    }

    function wordSection(m, tab, boxEl, key, note, single) {
        var label = labelsOf(boxEl, key)[0];
        var word = label ? label.dataset.word : key;
        var h = document.createElement("div");
        h.className = "wm-mh";
        var sw = document.createElement("i");
        sw.style.background = label ? getComputedStyle(label).getPropertyValue("--wm-c") : "#888";
        h.appendChild(sw);
        h.appendChild(document.createTextNode(word));
        if (note) {
            var sm = document.createElement("small");
            sm.textContent = note;
            h.appendChild(sm);
        }
        m.appendChild(h);
        var row = document.createElement("div");
        row.className = "wm-mrow";
        [["ignore ×0", 0], ["×0.5", 0.5], ["×1.5", 1.5], ["×2", 2], ["push away ×-0.5", -0.5]].forEach(function(f) {
            item(row, f[0], function() { send(tab, "set", word, f[1]); }, true);
        });
        item(row, "reset ×1", function() { send(tab, "reset", word); }, true);
        m.appendChild(row);
        if (single && label) {
            item(m, "👁 Show only this word", function() { label.click(); });
        }
        item(m, isLocked(tab, word) ? "🔓 Unlock (Optimize may change it)" : "🔒 Lock (Optimize leaves it alone)",
             function() { send(tab, "lock", word); });
        if (!(label && label.classList.contains("wm-x"))) {
            item(m, "✂ Remove from prompt", function() { removeFromPrompt(tab, word, boxEl); });
        }
        // words sharing its area (data-rel "word:overlap,..."): layers and new rules
        var rel = (label && label.dataset.rel || "").split(",").filter(Boolean).map(function(p) {
            var i = p.lastIndexOf(":");
            return [p.slice(0, i), p.slice(i + 1)];
        });
        var head = document.createElement("div");
        head.className = "wm-mi";
        head.style.cursor = "default";
        head.textContent = "📍 Move to (layer, next generation):";
        m.appendChild(head);
        var places = document.createElement("div");
        places.className = "wm-mrow";
        ["background", "foreground", "left", "right", "top", "bottom", "center", "skin", "clothes", "hair", "face", "body"].concat(
            rel.map(function(o) { return "on " + o[0]; })).concat(["none"]).forEach(function(t) {
            item(places, t === "none" ? "✖ no layer" : t, function() { sendRule(tab, "place", word, t); }, true);
        });
        m.appendChild(places);
        var head2 = document.createElement("div");
        head2.className = "wm-mi";
        head2.style.cursor = "default";
        head2.textContent = "📌 Always move to (place rule, every prompt with it):";
        m.appendChild(head2);
        var always = document.createElement("div");
        always.className = "wm-mrow";
        ["background", "foreground", "left", "right", "top", "bottom", "center", "skin", "clothes", "hair", "face", "body"].concat(
            rel.map(function(o) { return "on " + o[0]; })).forEach(function(t) {
            item(always, t, function() { sendRule(tab, "place_rule", word, t); }, true);
        });
        m.appendChild(always);
        [["↔ allow overlap with", "allow"], ["⭐ prefer over", "prefer"]].forEach(function(r) {
            if (!rel.length) return;
            var head = document.createElement("div");
            head.className = "wm-mi";
            head.style.cursor = "default";
            head.textContent = r[0] + " (rule):";
            m.appendChild(head);
            var chips = document.createElement("div");
            chips.className = "wm-mrow";
            rel.forEach(function(o) {
                item(chips, o[0] + " " + o[1], function() {
                    sendRule(tab, r[1], word, o[0]);
                    if (r[1] === "allow") forgetRelation(boxEl, word, o[0]);  // the next menu is right at once
                }, true);
            });
            m.appendChild(chips);
        });
    }

    function undoItem(m, tab) {
        var u = undoPrompt[tab];
        if (!u) return;
        m.appendChild(document.createElement("hr"));
        item(m, "↩ Undo: put '" + u.word + "' back into the prompt", function() { undoRemove(tab); });
    }

    function openMenu(e, build) {
        closeMenu();
        menuStyle();
        menu = document.createElement("div");
        menu.className = "wm-menu";
        build(menu);
        document.body.appendChild(menu);
        menu.style.left = Math.max(4, Math.min(e.clientX, window.innerWidth - menu.offsetWidth - 4)) + "px";
        menu.style.top = Math.max(4, Math.min(e.clientY, window.innerHeight - menu.offsetHeight - 4)) + "px";
    }

    document.addEventListener("contextmenu", function(e) {
        if (!e.target.closest) return;
        var label = e.target.closest(".wm-w[data-key]");
        var img = !label && e.target.closest(".wm-box img.wm-base");
        if (!label && !img) return;
        e.preventDefault();
        var boxEl = (label || img).closest(".wm-box");
        var tab = tabOf(boxEl);
        if (label) {
            openMenu(e, function(m) {
                wordSection(m, tab, boxEl, label.dataset.key, "", true);
                undoItem(m, tab);
            });
            return;
        }
        var f = spotOf(img, e);
        var hits = showSpot(img, f[0], f[1], false);
        openMenu(e, function(m) {
            if (!hits.length) {
                var d = document.createElement("div");
                d.className = "wm-mh";
                d.textContent = "No word stands out at this spot";
                m.appendChild(d);
            }
            hits.slice(0, 4).forEach(function(h, i) {
                if (i) m.appendChild(document.createElement("hr"));
                wordSection(m, tab, boxEl, h[0], h[1].toFixed(1) + "x here", false);
            });
            undoItem(m, tab);
        });
    }, true);

    document.addEventListener("click", function(e) {
        if (menu && !menu.contains(e.target)) closeMenu();
        if (!e.target.closest) return;
        var img = e.target.closest(".wm-box img.wm-base");
        if (img) {
            var f = spotOf(img, e);
            showSpot(img, f[0], f[1], true);
            return;
        }
        var label = e.target.closest(".wm-w[data-word]");
        if (label) send(tabOf(label), "select", label.dataset.word);
    }, true);

    document.addEventListener("keydown", function(e) {
        if (e.key === "Escape") closeMenu();
    });
    // scrolling the page closes the menu, scrolling inside it does not
    window.addEventListener("scroll", function(e) {
        if (menu && e.target instanceof Node && menu.contains(e.target)) return;
        closeMenu();
    }, true);
    window.addEventListener("resize", closeMenu);

    // ---- refresh -------------------------------------------------------------------------

    function refresh(tab) {
        var btn = gradioApp().querySelector("#act_steer_" + tab + "_show");
        if (btn) btn.click();
    }

    // a finished job raises the server's counter: reload the word map
    var seen = {txt2img: null, img2img: null};
    function poll() {
        if (!gradioApp().querySelector("#act_steer_txt2img_show, #act_steer_img2img_show")) return;
        fetch("./actsteer/version", {cache: "no-store"}).then(function(r) {
            return r.ok ? r.json() : null;
        }).then(function(v) {
            if (!v) return;
            ["txt2img", "img2img"].forEach(function(tab) {
                if (seen[tab] !== v[tab]) {
                    var first = seen[tab] === null;
                    seen[tab] = v[tab];
                    if (!first || v[tab] > 0) refresh(tab);
                    // auto optimize: measure the new image, maybe optimise and generate again
                    // (does nothing on the server when auto mode is not running)
                    var step = gradioApp().querySelector("#act_steer_" + tab + "_autostep");
                    if (!first && step) step.click();
                }
            });
        }).catch(function() {});
    }
    setInterval(poll, 1500);

    document.addEventListener("click", function(e) {
        var thumb = e.target.closest && e.target.closest("#txt2img_gallery .thumbnail-item, #img2img_gallery .thumbnail-item");
        if (!thumb) return;
        var tab = thumb.closest("#img2img_gallery") ? "img2img" : "txt2img";
        setTimeout(function() { refresh(tab); }, 150);
    }, true);
})();
