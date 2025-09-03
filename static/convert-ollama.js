const fs = require("fs");

const ollama = JSON.parse(fs.readFileSync("ollama.json", "utf8"));

const llms = [];
const embs = [];
ollama.list.forEach(x => {
    const mainTags = x.mainTags.map(x => x.includes(":latest") ? x.replace(":latest", "") : x);
    if (x.capabilities.includes("embedding")) {
        embs.push(mainTags);
    }
    else {
        llms.push(mainTags);
    }
});
fs.writeFileSync("ollama-min.json", JSON.stringify({embeddings: embs.flat(), llms: llms.flat()}, null, 2), "utf8");