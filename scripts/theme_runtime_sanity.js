const fs = require("fs");
const assert = require("assert");

const html = fs.readFileSync("app/templates/index.html", "utf8");
const tokens = fs.readFileSync("app/static/css/tokens.css", "utf8");
const theme = fs.readFileSync("app/static/js/theme.js", "utf8");
const shell = fs.readFileSync("app/static/js/ui-shell.js", "utf8");

assert(html.indexOf('/static/js/theme.js') < html.indexOf('/static/css/app.css'), "theme boot must run before CSS to prevent a color flash");
for (const mode of ["system", "light", "dark"]) assert(html.includes(`value="${mode}"`), `missing ${mode} theme option`);
assert(theme.includes('localStorage.getItem(STORAGE_KEY)'), "theme choice must persist");
assert(theme.includes('prefers-color-scheme: dark'), "system theme must follow OS preference");
assert(theme.includes('dataset.resolvedTheme'), "theme must expose its resolved light/dark state");
assert(tokens.includes(':root[data-resolved-theme="dark"]'), "dark mode token override is missing");
assert(shell.includes('data-route="home"') || html.includes('data-route="home"'), "Home route is missing");
assert(shell.includes('setLandingMode'), "Home and Import must have one explicit route controller");
assert(shell.includes('shellMounted'), "shell event installation must be idempotent");
assert(!shell.includes('document.addEventListener("click", (event) => {\n      document.querySelectorAll'), "global disclosure click sweeper must not return");
console.log("theme runtime sanity: PASS");
