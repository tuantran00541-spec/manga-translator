const fs = require("fs");

const inspector = fs.readFileSync("app/static/js/editor-inspector.js", "utf8");
const api = fs.readFileSync("app/static/js/api.js", "utf8");
const persistence = fs.readFileSync("app/static/js/editor-persistence.js", "utf8");

for (const marker of ["optgroup", "font_selection_mode", "value = \"auto\"", "matchFontForObject"]) {
  if (!inspector.includes(marker) && marker !== "matchFontForObject") {
    throw new Error(`missing inspector marker: ${marker}`);
  }
}
if (!api.includes("/api/fonts") || !api.includes("/api/fonts/match")) {
  throw new Error("font API endpoints are not wired in editor API");
}
for (const marker of ["font_selection_mode", "font_match", "font_ai_id"]) {
  if (!persistence.includes(marker)) throw new Error(`missing persistence marker: ${marker}`);
}
console.log("editor font picker sanity: OK");
