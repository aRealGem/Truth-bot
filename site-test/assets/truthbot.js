/* truth-bot minimal JS — expand/collapse only */
(function() {
  document.querySelectorAll('.expand-btn').forEach(function(btn) {
    btn.addEventListener('click', function() {
      var targetId = btn.getAttribute('data-target');
      var el = document.getElementById(targetId);
      if (!el) return;
      var open = el.classList.toggle('open');
      btn.textContent = open ? (btn.getAttribute('data-close') || 'Hide') : btn.getAttribute('data-open');
    });
  });
})();
