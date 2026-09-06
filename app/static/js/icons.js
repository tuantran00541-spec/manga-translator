(() => {
  const SVG_NS = "http://www.w3.org/2000/svg";

  window.createUiIcon = (name, className = "") => {
    const icon = document.createElementNS(SVG_NS, "svg");
    icon.setAttribute("aria-hidden", "true");
    icon.setAttribute("focusable", "false");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.classList.add("ui-icon");
    if (className) icon.classList.add(...className.split(" ").filter(Boolean));
    const use = document.createElementNS(SVG_NS, "use");
    use.setAttribute("href", `/static/icons.svg#${name}`);
    icon.appendChild(use);
    return icon;
  };
})();
