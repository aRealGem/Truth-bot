/* ─────────────────────────────────────────────────────────────────────
   truthbot.js — Truthy McTruthface state machine + Web Audio droid sounds

   Reads two attributes from #truthy-mascot-widget:
     data-mood        : 'happy' | 'iffy' | 'sad'   (computed by the pipeline)
     data-claim-count : integer; if 1, uses singular "that" wording,
                        otherwise uses aggregate "this" wording

   Updates #truthy-bubble text to match mood + count.
   Click (or Enter/Space when focused) plays the appropriate droid sound.

   No dependencies. Safe to load at the bottom of <body> or in <head>
   (DOMContentLoaded wrapper handles either case).
   ───────────────────────────────────────────────────────────────────── */

(function() {
  'use strict';

  function init() {
    var mascot       = document.getElementById('mascot');
    var widget       = document.getElementById('truthy-mascot-widget');
    if (!mascot || !widget) return;  // graceful no-op if Truthy isn't on this page

    var led          = document.getElementById('led');
    var ledHalo      = document.getElementById('ledHalo');
    var eyeLeftGroup = document.getElementById('eyeLeftGroup');
    var eyeRightGroup= document.getElementById('eyeRightGroup');
    var headGroup    = document.getElementById('headGroup');
    var bodyGroup    = document.getElementById('bodyGroup');
    var armLeftSwing = document.getElementById('armLeftSwing');
    var armRightSwing= document.getElementById('armRightSwing');
    var clipboard    = document.getElementById('clipboard');
    var bubble       = document.getElementById('truthy-bubble');

    /* ─── Captions: claim-count-aware ─── */
    var captionsSingle = {
      true: "That checks out. Sources match!",
      iffy: "Hmm… let me double-check my sources.",
      lie:  "Oh no… that isn't true."
    };
    var captionsMulti = {
      true: "All sources check out. Looking good!",
      iffy: "Mixed signals — some hold up, some don't.",
      lie:  "Oh no… most of this doesn't check out."
    };
    function getCaption(state, count) {
      return (count === 1 ? captionsSingle : captionsMulti)[state] || "";
    }
    var bubbleClassMap = { true: 'is-true', iffy: 'is-iffy', lie: 'is-lie' };

    var claimCount = parseInt(widget.getAttribute('data-claim-count'), 10);
    if (isNaN(claimCount)) claimCount = 0;  // 0 → uses multi-claim phrasing

    /* ─── State setter ─── */
    function setState(state) {
      mascot.classList.remove('state-true', 'state-iffy', 'state-lie');
      mascot.classList.add('state-' + state);

      if (bubble) {
        bubble.textContent = getCaption(state, claimCount);
        bubble.classList.remove('is-true', 'is-iffy', 'is-lie');
        bubble.classList.add(bubbleClassMap[state]);
      }

      if (state === 'true') {
        led.setAttribute('fill', 'url(#ledGradTrue)');
        ledHalo.setAttribute('fill', '#5ac075');
        eyeLeftGroup.setAttribute('transform', 'translate(115 154) rotate(0)');
        eyeRightGroup.setAttribute('transform', 'translate(185 154) rotate(0)');
        headGroup.setAttribute('transform', 'translate(0,0)');
        bodyGroup.setAttribute('transform', 'translate(0,0)');
        armLeftSwing.setAttribute('transform', 'rotate(135 88 253)');
        armRightSwing.setAttribute('transform', 'rotate(-135 212 253)');
        if (clipboard) clipboard.setAttribute('transform', 'translate(228 218) rotate(-8)');
      } else if (state === 'iffy') {
        led.setAttribute('fill', 'url(#ledGradIffy)');
        ledHalo.setAttribute('fill', '#e8b850');
        eyeLeftGroup.setAttribute('transform', 'translate(115 156) rotate(-10)');
        eyeRightGroup.setAttribute('transform', 'translate(185 156) rotate(10)');
        headGroup.setAttribute('transform', 'rotate(-7 150 170)');
        bodyGroup.setAttribute('transform', 'translate(0,0)');
        armLeftSwing.setAttribute('transform', 'rotate(0 88 253)');
        armRightSwing.setAttribute('transform', 'rotate(-110 212 253)');
        if (clipboard) clipboard.setAttribute('transform', 'translate(238 224) rotate(-3)');
      } else if (state === 'lie') {
        led.setAttribute('fill', 'url(#ledGradLie)');
        ledHalo.setAttribute('fill', '#5a8ec0');
        eyeLeftGroup.setAttribute('transform', 'translate(115 170) rotate(0)');
        eyeRightGroup.setAttribute('transform', 'translate(185 170) rotate(0)');
        headGroup.setAttribute('transform', 'translate(0,7)');
        bodyGroup.setAttribute('transform', 'translate(0,3)');
        armLeftSwing.setAttribute('transform', 'rotate(8 88 253)');
        armRightSwing.setAttribute('transform', 'rotate(35 212 253)');
        if (clipboard) clipboard.setAttribute('transform', 'translate(174 298) rotate(40)');
      }
    }

    /* ─── Idle blink scheduler ─── */
    function doBlink() {
      mascot.classList.add('blinking');
      setTimeout(function() { mascot.classList.remove('blinking'); }, 110);
    }
    function scheduleBlink() {
      var d = 2500 + Math.random() * 4500;
      setTimeout(function() {
        doBlink();
        if (Math.random() < 0.2) setTimeout(doBlink, 280);  // 20% chance of double-blink
        scheduleBlink();
      }, d);
    }
    scheduleBlink();

    /* ─── Web Audio droid sounds ─────────────────────────────────────
       Synthesized via Web Audio API. No audio files needed,
       no licensing, no network round-trips. All sounds resolve in
       <500ms.

       Autoplay-policy contract: browsers (especially Safari) leave a
       freshly-created AudioContext in ``suspended`` until a user
       gesture explicitly resumes it. ``audioCtx.resume()`` returns
       a Promise. The earlier implementation called resume() and
       *immediately* scheduled oscillators against ``ctx.currentTime``
       — on Safari and some Chrome variants the context was still
       suspended at schedule time, so the oscillator silently
       no-op'd. The fix: ``unlockAudio()`` returns a Promise, and the
       play functions are only invoked after that Promise resolves.
       ──────────────────────────────────────────────────────────── */
    var audioCtx = null;
    function unlockAudio() {
      if (!audioCtx) {
        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) { return Promise.resolve(null); }
      }
      if (audioCtx.state === 'suspended') {
        var p = audioCtx.resume();
        // Some old Safari versions return undefined from resume().
        if (p && typeof p.then === 'function') {
          return p.then(function() { return audioCtx; },
                        function() { return audioCtx; });
        }
      }
      return Promise.resolve(audioCtx);
    }

    // Happy: bright rising arpeggio (C5 → E5 → G5 → C6) with square wave
    function playHappy(ctx) {
      var notes = [523.25, 659.25, 783.99, 1046.50];
      notes.forEach(function(freq, i) {
        var t0 = ctx.currentTime + i * 0.07;
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'square';
        osc.frequency.setValueAtTime(freq, t0);
        gain.gain.setValueAtTime(0, t0);
        gain.gain.linearRampToValueAtTime(0.12, t0 + 0.01);
        gain.gain.linearRampToValueAtTime(0, t0 + 0.10);
        osc.connect(gain).connect(ctx.destination);
        osc.start(t0);
        osc.stop(t0 + 0.12);
      });
    }

    // Confused: triangle wave bending up to ~620Hz then dropping to ~330Hz
    function playConfused(ctx) {
      var t0 = ctx.currentTime;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(440, t0);
      osc.frequency.exponentialRampToValueAtTime(620, t0 + 0.18);
      osc.frequency.exponentialRampToValueAtTime(330, t0 + 0.42);
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(0.14, t0 + 0.02);
      gain.gain.linearRampToValueAtTime(0, t0 + 0.45);
      osc.connect(gain).connect(ctx.destination);
      osc.start(t0);
      osc.stop(t0 + 0.5);
    }

    // Sad: descending minor third (G4 → Eb4) with downward pitch bend on each note
    function playSad(ctx) {
      var notes = [392.00, 311.13];
      notes.forEach(function(freq, i) {
        var t0 = ctx.currentTime + i * 0.20;
        var osc = ctx.createOscillator();
        var gain = ctx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, t0);
        osc.frequency.linearRampToValueAtTime(freq * 0.93, t0 + 0.25);
        gain.gain.setValueAtTime(0, t0);
        gain.gain.linearRampToValueAtTime(0.15, t0 + 0.03);
        gain.gain.linearRampToValueAtTime(0, t0 + 0.28);
        osc.connect(gain).connect(ctx.destination);
        osc.start(t0);
        osc.stop(t0 + 0.32);
      });
    }

    var soundMap = { true: playHappy, iffy: playConfused, lie: playSad };

    /* ─── Speak handler ──────────────────────────────────────────────
       Awaits the AudioContext unlock Promise before scheduling
       oscillators. Browsers that silently dropped the prior
       fire-and-forget pattern now actually emit sound.
       ──────────────────────────────────────────────────────────── */
    function speak() {
      var match = mascot.className.match(/state-(true|iffy|lie)/);
      if (!match) return;
      var state = match[1];
      var fn = soundMap[state];
      if (!fn) return;
      mascot.classList.add('speaking');
      setTimeout(function() { mascot.classList.remove('speaking'); }, 700);
      unlockAudio().then(function(ctx) { if (ctx) fn(ctx); });
    }

    /* ─── Initialize ─── */
    var mood = widget.getAttribute('data-mood') || 'iffy';
    var stateMap = { happy: 'true', iffy: 'iffy', sad: 'lie' };
    setState(stateMap[mood] || 'iffy');

    /* ─── Site-wide mute state + queued first-gesture autoplay ─────
       Default: ``mute === 'off'`` (sound enabled). On report and
       index pages we attempt a one-shot mood sound on the user's
       first interaction with the page (browser autoplay policies
       block AudioContext.start() until a gesture). On the dedicated
       Truthy fun page we keep the legacy "tap = always plays"
       behavior so the page stays a playground.

       Persistence: localStorage["truthy-mute"] in {"on", "off"}.
       ─────────────────────────────────────────────────────────── */
    var TRUTHY_MUTE_KEY = 'truthy-mute';
    var DEFAULT_TRUTHY_MUTE = 'off';
    var path = (window.location && window.location.pathname) || '';
    /* The dedicated Truthy fun page keeps the legacy "tap always plays"
       behavior; everywhere else uses the mute toggle. Detection is by
       URL path substring so query strings / hashes don't trip it up. */
    var isTruthyFunPage = path.indexOf('truthy.html') !== -1;

    function readMute() {
      try {
        var v = localStorage.getItem(TRUTHY_MUTE_KEY);
        return (v === 'on' || v === 'off') ? v : DEFAULT_TRUTHY_MUTE;
      } catch (e) { return DEFAULT_TRUTHY_MUTE; }
    }
    function writeMute(v) {
      try { localStorage.setItem(TRUTHY_MUTE_KEY, v); } catch (e) { /* ignore */ }
    }

    var tapHintLabel = widget.querySelector('.tap-hint-label');
    function updateTapHintLabel(mute) {
      if (!tapHintLabel) return;
      if (isTruthyFunPage) {
        tapHintLabel.textContent = 'Tap';
      } else if (mute === 'on') {
        tapHintLabel.textContent = 'Muted';
      } else {
        tapHintLabel.textContent = 'Tap to mute';
      }
    }
    if (tapHintLabel) widget.setAttribute('data-mute', isTruthyFunPage ? 'na' : readMute());
    updateTapHintLabel(readMute());

    /* Queued first-gesture autoplay. Suppressed on the fun page
       (legacy behavior). Removed if the user explicitly taps the
       mascot before any other gesture (taking explicit control of
       the mute toggle should not also fire the queued play).

       ``pointerdown`` fires *before* the subsequent ``click``, which
       matters when the user's first gesture is on a navigation link:
       click navigates the page away, while pointerdown gives the
       AudioContext unlock + oscillator schedule a head start. */
    var queuedHandler = null;
    var QUEUE_EVENTS = ['pointerdown', 'click', 'keydown', 'touchstart'];
    function removeQueued() {
      if (!queuedHandler) return;
      QUEUE_EVENTS.forEach(function(evt) {
        document.removeEventListener(evt, queuedHandler, true);
      });
      queuedHandler = null;
    }
    function setupQueuedAutoplay() {
      if (isTruthyFunPage) return;
      if (readMute() === 'on') return;
      queuedHandler = function() { removeQueued(); speak(); };
      QUEUE_EVENTS.forEach(function(evt) {
        document.addEventListener(evt, queuedHandler, true);
      });
    }
    setupQueuedAutoplay();

    function onMascotActivate(e) {
      if (isTruthyFunPage) {
        speak();
        return;
      }
      /* User explicitly took control before any queued autoplay
         could fire — cancel it so the click only does the mute
         toggle, not also a play. */
      removeQueued();
      if (e && e.stopPropagation) e.stopPropagation();
      var current = readMute();
      var next = (current === 'on') ? 'off' : 'on';
      writeMute(next);
      widget.setAttribute('data-mute', next);
      updateTapHintLabel(next);
      if (next === 'off') speak();  // unmuting always plays once
    }

    widget.addEventListener('click', onMascotActivate);
    widget.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        onMascotActivate(e);
      }
    });
  }

  // Run init immediately if DOM is already parsed; otherwise wait
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ─────────────────────────────────────────────────────────────────────
   Editorial-lens toggle — flips every Truthy-scale display between the
   Lenient (default) and Strict 5-bucket coarse-axis projections.

   Two render patterns are toggled together so the page never goes
   internally inconsistent (e.g. headline says "Mostly Truthy" while the
   verdict bar still shows the Strict aggregate):

   1) PER-PILL SWAP — in-place text+class rewrite on individual pills.
      Used by the per-claim headline pill (claim card) and the
      per-claim TOC mini-pill on report pages. Both wear ``.lens-pill``
      and carry the data-coarse-{lenient,strict} attribute pair.

   2) PAIRED-AXIS SWAP — show/hide complementary blocks pre-rendered
      server-side. Used by aggregate views: the verdict-panel headline
      + ratio + bar, the per-report cards on the index, and any future
      lens-aware aggregate. Each block wears ``[data-lens-axis="X"]``
      and the toggle simply flips the ``hidden`` attribute.

   The per-model strip pills (Anthropic / OpenAI / Gemini / xAI) are
   NEVER touched — they keep the 6-bucket fine labels for audit.

   Body data attribute ``document.body.dataset.lens`` is also set so
   any lens-aware CSS rule can react.

   Persistence: ``localStorage.editorial-lens`` ∈ {"lenient","strict"}.
   Default: strict (2026-04-30 editorial flip from Lenient — Strict
   tracks more closely with the reference set per FitnessScorer Run 5
   and stays the conservative default for non-JS clients). Stored
   user preference still wins on revisit.
   No-op if the page has nothing toggleable (e.g. about, 404).
   ───────────────────────────────────────────────────────────────────── */
(function() {
  'use strict';

  var STORAGE_KEY = 'editorial-lens';
  var DEFAULT_LENS = 'strict';
  var ALL_PILL_CSS_CLASSES = [
    'v-true', 'v-mostly-true', 'v-exaggerated', 'v-misleading',
    'v-false', 'v-unverifiable', 'v-truthy', 'v-falsey'
  ];

  function readLens() {
    try {
      var v = localStorage.getItem(STORAGE_KEY);
      return (v === 'strict' || v === 'lenient') ? v : DEFAULT_LENS;
    } catch (e) {
      return DEFAULT_LENS;
    }
  }

  function writeLens(lens) {
    try { localStorage.setItem(STORAGE_KEY, lens); } catch (e) { /* ignore */ }
  }

  function applyLensToPill(pill, lens) {
    var label, cssSlug;
    if (lens === 'strict') {
      label = pill.getAttribute('data-coarse-strict') || pill.getAttribute('data-fine-label') || '';
      cssSlug = pill.getAttribute('data-coarse-strict-css') || pill.getAttribute('data-fine-css') || 'unverifiable';
    } else {
      label = pill.getAttribute('data-coarse-lenient') || pill.getAttribute('data-fine-label') || '';
      cssSlug = pill.getAttribute('data-coarse-lenient-css') || pill.getAttribute('data-fine-css') || 'unverifiable';
    }
    if (!label) return;
    pill.textContent = label;
    for (var i = 0; i < ALL_PILL_CSS_CLASSES.length; i++) {
      pill.classList.remove(ALL_PILL_CSS_CLASSES[i]);
    }
    pill.classList.add('v-' + cssSlug);
  }

  function applyLensToAxisPairs(lens) {
    /* Show the block tagged with the active lens, hide the other.
       Idempotent — safe to call repeatedly. */
    var blocks = document.querySelectorAll('[data-lens-axis]');
    for (var i = 0; i < blocks.length; i++) {
      var axis = blocks[i].getAttribute('data-lens-axis');
      if (axis === lens) {
        blocks[i].hidden = false;
      } else {
        blocks[i].hidden = true;
      }
    }
  }

  function applyLens(lens) {
    /* 1) per-pill text+class swap (headline pill + TOC pill) */
    var pills = document.querySelectorAll('.lens-pill');
    for (var i = 0; i < pills.length; i++) {
      applyLensToPill(pills[i], lens);
    }
    /* 2) paired-axis show/hide for aggregate displays */
    applyLensToAxisPairs(lens);
    /* 3) body data-attr so any lens-aware CSS rule can react */
    if (document.body) document.body.setAttribute('data-lens', lens);
    /* 4) chip state */
    var chip = document.querySelector('.editorial-lens');
    if (chip) {
      chip.setAttribute('data-lens', lens);
      var valEl = chip.querySelector('.lens-value');
      if (valEl) valEl.textContent = (lens === 'strict') ? 'Strict' : 'Lenient';
      chip.setAttribute('aria-pressed', lens === 'strict' ? 'true' : 'false');
    }
  }

  function init() {
    var pills = document.querySelectorAll('.lens-pill');
    var axisBlocks = document.querySelectorAll('[data-lens-axis]');
    var chip = document.querySelector('.editorial-lens');
    var hasToggleableContent = pills.length > 0 || axisBlocks.length > 0;
    if (!hasToggleableContent) {
      if (chip) chip.hidden = true;
      return;
    }
    var lens = readLens();
    applyLens(lens);
    if (chip) {
      chip.hidden = false;
      chip.addEventListener('click', function() {
        var current = chip.getAttribute('data-lens') || DEFAULT_LENS;
        var next = (current === 'lenient') ? 'strict' : 'lenient';
        writeLens(next);
        applyLens(next);
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

