const root = document.getElementById("root");

function renderHtml(html) {
    const div = document.createElement("div");
    div.innerHTML = html;
    return div.children[0];
}

function showHtml(html) {
    root.innerHTML = html;
}

async function showServicesPage() {
    showLoadingPage();
    const data = await (await fetchX("/admin/services")).json();
    const html = `
        <div class="page services-page">
            <h3>Services</h3>
            <div class="boxes services">
                ${data.list.map(s => `
                    <div class="box service">
                        <div class="box-name service-name">
                            ${s.id}
                            <div class="badge ${s.installed ? "badge-installed" : "badge-not-insalled"}">
                                ${s.installed ? "Installed" : "Not installed"}
                            </div>
                        </div>
                        <div class="box-buttons service-buttons">
                            ${!s.installed ? `<button data-action="install-service" data-service-id="${s.id}">Install</button>` : ""}
                            ${s.installed ? `<button data-action="open-service-models" data-service-id="${s.id}">Models</button>`: ""}
                            ${s.installed ? `<button data-action="uninstall-service" data-service-id="${s.id}">Uninstall</button>`: ""}
                        </div>
                    </div>`).join("")}
            </div>
        </div>`;
    showHtml(html);
}

let modelFilter = "";
let modelType = "__all";
let modelInstalled = "__all";

function resetModelFilters() {
    modelFilter = "";
    modelType = "__all";
    modelInstalled = "__all";
}

function showInstallServiceModal(info, onInstall) {
    const html = `
<div class="modal-backdrop">
    <div class="modal">
        <div class="modal-title">Install ${info.id}</div>
        <form class="fields" data-id="modal-form">
            ${info.spec.fields.map(field => `
                <div class="field">
                    <div class="field-label">${field.description}</div>
                    <div class="field-input">${(() => {
                        if (field.type === "text" || field.type === "password" || field.type === "number") {
                            return `<input type="${field.type}" name="${field.name}" required="required" value="${field.default || ""}" placeholder="${field.description}" />`;
                        }
                        if (field.type === "bool") {
                            return `<input type="checkbox" name="${field.name}" />`;
                        }
                    })()}</div>
                </div>
            `).join("")}
        </form>
        <div class="buttons">
            <button data-id="cancel">Cancel</button>
            <button data-id="install">Install</button>
        </div>
    </div>
</div>`;
    const div = document.createElement("div");
    div.innerHTML = html;
    document.body.append(div);
    div.querySelector("[data-id=cancel]").addEventListener("click", () => {
        div.remove();
    });
    div.querySelector("[data-id=install]").addEventListener("click", () => {
        const valid = div.querySelector("form").reportValidity();
        if (!valid) {
            return;
        }
        const data = {};
        for (const field of div.querySelectorAll("input")) {
            data[field.name] = field.type == "checkbox" ? field.checked : field.value;
        }
        onInstall(data);
        div.remove();
    });
}

async function showServicePage(id) {
    showLoadingPage();
    const data = await (await fetchX(`/admin/services/${id}/models`)).json();
    const list = data.list.sort((a, b) => {
        if (a.installed !== b.installed) {
            return a.installed ? -1 : 1;
        }
        if (a.type !== b.type) {
            return a.type.localeCompare(b.type);
        }
        return a.id.localeCompare(b.id);
    });
    const types = {
        tts: "Text to speech (TTS)",
        stt: "Speech to text (STT)",
        llm: "Large Language Model (LLM)",
        embedding: "Embedding (embedding)"
    };
    const installedOptions = {
        __all: null,
        installed: true,
        notinstalled: false
    };
    const html = `
        <div class="page service-page">
            <h3>
                Service: ${id}
                <button data-action="open-services">Back to services</button>
            </h3>
            <div class="search">
                <input id="model-filter" type="text" placeholder="Filter by name..." />
            </div>
            <div style="margin: 20px 0 0 0">
                <span>Type:</span>
                <select id="model-type">
                    <option value="__all">--ALL--</option>
                    ${Object.keys(types).map(t =>
                        `<option value="${t}">${types[t]}</option>`
                    ).join("")}
                </select>
                
                <span style="margin-left: 20px;">Installed:</span>
                <select id="model-installed">
                    <option value="__all">--ALL--</option>
                    <option value="installed">Only installed</option>
                    <option value="notinstalled">Only not installed</option>
                </select>
            </div>
            <div class="boxes models">
            </div>
        </div>`;
    showHtml(html);
    const boxes = document.querySelector(".boxes.models");
    function render() {
        const filterl = modelFilter.toLowerCase();
        const installed = installedOptions[modelInstalled];
        const filtered = list.filter(x => 
            (!filterl || x.id.toLowerCase().includes(filterl)) &&
            (modelType === "__all" || x.type === modelType) &&
            (installed === null || x.installed === installed)
        );
        boxes.innerHTML = filtered.length === 0 ? `<div class="empty">No elements</div>` : filtered.map(m => `
            <div class="box model">
                <div class="box-name model-name">
                    ${m.id}
                    <div class="badge ${m.installed ? "badge-installed" : "badge-not-insalled"}">
                        ${m.installed ? "Installed" : "Not installed"}
                    </div>
                </div>
                <div class="model-type">${types[m.type] || m.type}</div>
                <div class="box-buttons model-buttons">
                    ${!m.installed ? `<button data-action="install-model" data-service-id="${id}" data-model-id="${m.id}">Install</button>` : ""}
                    ${m.installed ? `<button data-action="uninstall-model" data-service-id="${id}" data-model-id="${m.id}">Uninstall</button>`: ""}
                </div>
            </div>`).join("")
    }
    render();
    const modelFilterEle = document.getElementById("model-filter");
    modelFilterEle.value = modelFilter;
    modelFilterEle.addEventListener("input", () => {
        modelFilter = modelFilterEle.value;
        render();
    });
    const modelTypeEle = document.getElementById("model-type");
    modelTypeEle.value = modelType;
    modelTypeEle.addEventListener("change", () => {
        modelType = modelTypeEle.value;
        render();
    });
    const modelInstalledEle = document.getElementById("model-installed");
    modelInstalledEle.value = modelInstalled;
    modelInstalledEle.addEventListener("change", () => {
        modelInstalled = modelInstalledEle.value;
        render();
    });
}

function showLoadingPage() {
    showHtml(`<div class="loading">Loading...</div>`);
}

root.addEventListener("click", async e => {
    const dataActionEle = e.target.closest("[data-action]");
    if (dataActionEle) {
        const action = dataActionEle.getAttribute("data-action");
        const serviceId = dataActionEle.getAttribute("data-service-id");
        const modelId = dataActionEle.getAttribute("data-model-id");
        if (action === "install-service") {
            const info = await (await fetchX(`/admin/services/${serviceId}`, {
                method: "GET"
            })).json();
            showInstallServiceModal(info, async data => {
                showLoadingPage();
                try {
                    await (await fetchX(`/admin/services/${serviceId}`, {
                        method: "POST",
                        body: JSON.stringify({spec: data}),
                        headers: {"Content-Type": "application/json"}
                    })).json();
                }
                catch (e) {
                    console.log("Error", e);
                    alert("Error");
                }
                showServicesPage();
            });
        }
        else if (action === "uninstall-service") {
            showLoadingPage()
            await (await fetchX(`/admin/services/${serviceId}`, {
                method: "DELETE",
                body: JSON.stringify({purge: false}),
                headers: {"Content-Type": "application/json"}
            })).json();
            showServicesPage();
        }
        else if (action === "open-service-models") {
            resetModelFilters();
            showServicePage(serviceId);
        }
        else if (action === "install-model") {
            showLoadingPage()
            await (await fetchX(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                method: "POST",
                body: JSON.stringify({alias: null}),
                headers: {"Content-Type": "application/json"}
            })).json();
            showServicePage(serviceId);
        }
        else if (action === "uninstall-model") {
            showLoadingPage()
            await (await fetchX(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                method: "DELETE",
                body: JSON.stringify({purge: false}),
                headers: {"Content-Type": "application/json"}
            })).json();
            showServicePage(serviceId);
        }
        else if (action === "open-services") {
            showServicesPage();
        }
    }
});

const LOCAL_STORAGE_KEY_ADMIN_API_KEY = "admin_api_key";

async function fetchX(url, options) {
    const opts = options || {};
    opts.headers = opts.headers || {};
    opts.headers["Authorization"] = "Bearer " + admin_api_key;
    const response = admin_api_key ? await fetch(url, opts) : {status: 403};
    if (response.status === 401 || response.status === 403) {
        showAdminApiKeyModal();
        document.location.reload();
        throw new Error("Unauthorized");
    }
    return response;
}

function showAdminApiKeyModal() {
    const api_key = prompt("Type admin api key");
    if (api_key) {
        localStorage.setItem(LOCAL_STORAGE_KEY_ADMIN_API_KEY, api_key);
    }
}

const admin_api_key = localStorage.getItem(LOCAL_STORAGE_KEY_ADMIN_API_KEY);
showServicesPage()
