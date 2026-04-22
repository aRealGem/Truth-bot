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
       no licensing, no network round-trips.
       AudioContext is lazily created on first user gesture (browsers
       block autoplay otherwise). All sounds resolve in <500ms.
       ──────────────────────────────────────────────────────────── */
    var audioCtx = null;
    function getCtx() {
      if (!audioCtx) {
        try {
          audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        } catch (e) { return null; }
      }
      if (audioCtx && audioCtx.state === 'suspended') audioCtx.resume();
      return audioCtx;
    }

    // Happy: bright rising arpeggio (C5 → E5 → G5 → C6) with square wave
    function playHappy() {
      var ctx = getCtx(); if (!ctx) return;
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
    function playConfused() {
      var ctx = getCtx(); if (!ctx) return;
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
    function playSad() {
      var ctx = getCtx(); if (!ctx) return;
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

    /* ─── Speak handler ─── */
    function speak() {
      var match = mascot.className.match(/state-(true|iffy|lie)/);
      if (!match) return;
      var state = match[1];
      var fn = soundMap[state];
      if (fn) fn();
      mascot.classList.add('speaking');
      setTimeout(function() { mascot.classList.remove('speaking'); }, 700);
    }

    /* ─── Initialize ─── */
    var mood = widget.getAttribute('data-mood') || 'iffy';
    var stateMap = { happy: 'true', iffy: 'iffy', sad: 'lie' };
    setState(stateMap[mood] || 'iffy');

    widget.addEventListener('click', speak);
    widget.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        speak();
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
