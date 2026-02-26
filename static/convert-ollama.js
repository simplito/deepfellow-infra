const fs = require("fs");

const ollama = JSON.parse(fs.readFileSync("ollama.json", "utf8"));

const llms = [];
const embs = [];
ollama.list.forEach(model => {
    const mainTags = model.mainTags.map(mainTag => {
        const tag = model.tags.find(tag => tag.tag == mainTag);
        return {name: mainTag.includes(":latest") ? mainTag.replace(":latest", "") : mainTag, size: tag.size, hash: tag.hash ?? ""};
    });
    if (model.capabilities.includes("embedding")) {
        embs.push(mainTags);
    }
    else {
        llms.push(mainTags);
    }
});
fs.writeFileSync("ollama-min.json", JSON.stringify({embeddings: embs.flat(), llms: llms.flat()}, null, 2), "utf8");