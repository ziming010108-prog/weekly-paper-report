// js/lang-toggle.js
(function () {
  function setLang(lang) {
    document.documentElement.setAttribute("data-lang", lang);
    try { localStorage.setItem("wpr-lang", lang); } catch (e) {}
  }

  function getSavedLang() {
    try { return localStorage.getItem("wpr-lang"); } catch (e) { return null; }
  }

  function init() {
    var sel = document.getElementById("lang-select");
    var saved = getSavedLang();
    var lang = saved || "en";
    setLang(lang);

    if (sel) {
      sel.value = lang;
      sel.addEventListener("change", function () {
        setLang(sel.value);
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
