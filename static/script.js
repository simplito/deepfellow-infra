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
    const data = await fetchX("/admin/services");
    const html = `
        <div class="page services-page">
            <h3>Services</h3>
            <div class="boxes services">
                ${data.list.map(s => {
                    const values = s.installed ? s.installed : false;
                    return (
                        `<div class="box service">
                            <div class="box-name service-name">
                                ${s.id}
                                <div class="badge ${s.installed ? "badge-installed" : "badge-not-insalled"}">
                                    ${s.installed ? "Installed" : "Not installed"}
                                </div>
                            </div>
                            <div class="box-size service-size">
                                ${typeof(s.size) == "string" ?
                                    `Size: <span class="size">${s.size ? s.size : "N/A"}</span>` :
                                    Object.keys(s.size).map(key =>
                                        `<div class="box-size-entry">${key.toUpperCase()}: <span class="size">${s.size[key]}</span></div>`
                                    ).join("")
                                }
                            </div>
                            <div class="box-values">
                                ${values ? Object.keys(values).map(key => `<div class="box-value">${key} = ${s.spec.fields.find(x => x.name === key)?.type === "password" ? "*****" : (values[key] === null ? "" : values[key])}</div>`).join("") : ""}
                            </div>
                            <div class="box-buttons service-buttons">
                                ${!s.installed ? `<button data-action="install-service" data-service-id="${s.id}">Install</button>` : ""}
                                ${s.installed ? `<button data-action="open-service-models" data-service-id="${s.id}">Models</button>`: ""}
                                ${s.installed ? `<button data-action="uninstall-service" data-service-id="${s.id}">Uninstall</button>`: ""}
                            </div>
                        </div>`
                    );
                }).join("")}
            </div>
        </div>`;
    showHtml(html);
}

let modelFilter = "";
let modelType = "__all";
let modelInstalled = "__all";
let modelCustom = "__all"

function resetModelFilters() {
    modelFilter = "";
    modelType = "__all";
    modelInstalled = "__all";
    modelCustom = "__all";
}

function showFormModal(title, mainButtonLabel, spec, onInstall) {
    const html = `
<div class="modal-backdrop">
    <div class="modal">
        <div class="modal-title">${title}</div>
        <form class="fields" data-id="modal-form">
            ${spec.fields.map(field => `
                <div class="field" ${field.display ? `data-display="${field.display}"` : ""}>
                    <div class="field-label">${field.description} ${!field.required ? `<span class="optional">(optional)</span>` : ""}</div>
                    <div class="field-input">${(() => {
                        if (field.type === "text" || field.type === "password" || field.type === "number") {
                            return `<input type="${field.type}" name="${field.name}" ${field.required ? `required="required"` : ""} value="${field.default || ""}" placeholder="${field.placeholder || ""}" />`;
                        }
                        if (field.type === "bool") {
                            return `<input type="checkbox" name="${field.name}" ${field.default == "true" ? `checked="checked"` : ""} />`;
                        }
                        if (field.type === "oneof") {
                            return `<select name="${field.name}" ${field.required ? `required="required"` : ""}>
                                ${field.values.map(x => `<option value="${x}">${x}</option>`).join("")}
                            </select>`;
                        }
                        if (field.type === "list") {
                            return [0,1,2,3,4,5].map(() =>
                                `<input type="text" data-list="${field.name}" placeholder="${field.placeholder || ""}" />`
                            ).join("<br/>");
                        }
                        if (field.type === "map") {
                            return [0,1,2,3,4,5].map((i) => `
                                <input type="text" data-map="${field.name}" data-map-key="${i}" placeholder="${field.placeholder || ""}" style="width: 45%" />
                                <input type="text" data-map="${field.name}" data-map-value="${i}" placeholder="${field.placeholder || ""}" style="width: 45%" />
                            `).join("<br/>");
                        }
                    })()}</div>
                </div>
            `).join("")}
        </form>
        <div class="buttons">
            <button data-id="cancel">Cancel</button>
            <button data-id="install">${mainButtonLabel}</button>
        </div>
    </div>
</div>`;
    function refreshDisplay() {
        [...div.querySelectorAll("[data-display]")].forEach(field => {
            const dataDisplay = field.getAttribute("data-display");
            const values = dataDisplay.split("=");
            if (values.length != 2) {
                return;
            }
            const [name, value] = values;
            const ele = div.querySelector(`[name=${name}]`);
            if (!ele) {
                return;
            }
            field.style.display = ele.value === value ? "block" : "none";
        });
    }
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
        const keyMapping = {};
        const data = {};
        for (const field of div.querySelectorAll("input,select")) {
            if (field.name) {
                data[field.name] = field.type == "checkbox" ? field.checked : field.type == "number" ? field.valueAsNumber : field.value;
            }
            else if (field.hasAttribute("data-list") && field.value) {
                if (!data[field.getAttribute("data-list")]) {
                    data[field.getAttribute("data-list")] = []
                }
                data[field.getAttribute("data-list")].push(field.value)
            }
            else if (field.hasAttribute("data-map")) {
                if (field.hasAttribute("data-map-key") && field.value) {
                    if (!data[field.getAttribute("data-map")]) {
                        data[field.getAttribute("data-map")] = {}
                    }
                    if (!keyMapping[field.getAttribute("data-map")]) {
                        keyMapping[field.getAttribute("data-map")] = {}
                    }
                    data[field.getAttribute("data-map")][field.value] = "";
                    keyMapping[field.getAttribute("data-map")][field.getAttribute("data-map-key")] = field.value;
                }
                else if (field.hasAttribute("data-map-value")) {
                    const km = keyMapping[field.getAttribute("data-map")]
                    const key = km ? km[field.getAttribute("data-map-value")] : null;
                    if (km && key && data[field.getAttribute("data-map")]) {
                        data[field.getAttribute("data-map")][key] = field.value;
                    }
                }
            }
        }
        onInstall(data);
        div.remove();
    });
    [...div.querySelectorAll("input,select")].forEach(input => {
        input.addEventListener("change", refreshDisplay);
    });
}

function showInstallServiceModal(info, onInstall) {
    return showFormModal(`Install ${info.id}`, "Install", info.spec, onInstall);
}

function showInstallModelModal(info, onInstall) {
    return showFormModal(`Install ${info.id}`, "Install", info.spec, onInstall);
}

function showAddCustomModelModal(info, onInstall) {
    return showFormModal(`Add custom model for ${info.id}`, "Add custom model", info.custom_model_spec, onInstall);
}

function showContentModal(options) {
    const html = `
<div class="modal-backdrop">
    <div class="modal ${options.wide ? "modal-wide" : ""}">
        <div class="modal-title">${options.title}</div>
        ${options.pre === false ? `<div class="content"></div>` : `<pre class="content"></pre>`}
        <div class="buttons">
            <button data-id="ok">OK</button>
        </div>
    </div>
</div>`;
    const div = document.createElement("div");
    div.innerHTML = html;
    div.querySelector(".content").textContent = options.text;
    document.body.append(div);
    div.querySelector("[data-id=ok]").addEventListener("click", () => {
        div.remove();
    });
}

async function showServicePage(id) {
    showLoadingPage();
    const serivceInfo = await fetchX(`/admin/services/${id}`);
    const data = await fetchX(`/admin/services/${id}/models`);
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
        embedding: "Embedding (embedding)",
        lora: "LORA (lora)"
    };
    const installedOptions = {
        __all: null,
        installed: true,
        notinstalled: false
    };
    const customOptions = {
        __all: null,
        onlycustom: true,
        onlynotcustom: false
    };
    const html = `
        <div class="page service-page">
            <h3>
                Service: ${id}
                ${serivceInfo.custom_model_spec ? `<button data-service-id="${id}" data-action="add-custom-model">Add custom model</button> ` : ""}
                ${serivceInfo.has_docker ? `
                    <button data-service-id="${id}" data-action="show-docker-logs">Show docker logs</button>
                    <button data-service-id="${id}" data-action="show-docker-compose-file">Show docker compose file</button>
                    <button data-service-id="${id}" data-action="restart-docker">Restart docker</button>` : ""}
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
                
                ${serivceInfo.custom_model_spec ? `<span style="margin-left: 20px;">Custom:</span>
                <select id="model-custom">
                    <option value="__all">--ALL--</option>
                    <option value="onlycustom">Only custom</option>
                    <option value="onlynotcustom">Only not custom</option>
                </select>` : ""}
            </div>
            <div class="boxes models">
            </div>
        </div>`;
    showHtml(html);
    const boxes = document.querySelector(".boxes.models");
    function render() {
        const filterl = modelFilter.toLowerCase();
        const installed = installedOptions[modelInstalled];
        const custom = customOptions[modelCustom];
        const filtered = list.filter(x => 
            (!filterl || x.id.toLowerCase().includes(filterl)) &&
            (modelType === "__all" || x.type === modelType) &&
            (installed === null || x.installed === installed) &&
            (custom === null || (!!x.custom) === custom)
        );
        boxes.innerHTML = filtered.length === 0 ? `<div class="empty">No elements</div>` : filtered.map(m => {
            const values = m.installed ? m.installed.spec || {} : false;
            return (
                `<div class="box model">
                    <div class="box-name model-name">
                        ${m.id}
                        <div class="badge ${m.installed ? "badge-installed" : "badge-not-insalled"}">
                            ${m.installed ? "Installed" : "Not installed"}
                        </div>
                        ${m.custom ? `<div class="badge">
                            Custom
                        </div>`: ""}
                    </div>
                    <div class="model-type">${types[m.type] || m.type}</div>
                    <div class="box-size model-size">
                        Size: <span class="size">${m.size ? m.size : "N/A"}</span>
                    </div>
                    <div class="box-values">
                        ${values ? Object.keys(values).map(key => `<div class="box-value">${key} = ${m.spec.fields.find(x => x.name === key)?.type === "password" ? "*****" : values[key]}</div>`).join("") : ""}
                    </div>
                    <div class="box-buttons model-buttons">
                        ${!m.installed ? `<button data-action="install-model" data-service-id="${id}" data-model-id="${m.id}">Install</button>` : ""}
                        ${m.installed ? `<button data-action="uninstall-model" data-service-id="${id}" data-model-id="${m.id}">Uninstall</button>`: ""}
                        ${m.installed && m.has_docker ? `
                            <button data-service-id="${id}" data-model-id="${m.id}" data-action="show-docker-logs">Show docker logs</button>
                            <button data-service-id="${id}" data-model-id="${m.id}" data-action="show-docker-compose-file">Show docker compose file</button>
                            <button data-service-id="${id}" data-model-id="${m.id}" data-action="restart-docker">Restart docker</button>` : ""}
                        ${!m.installed && m.custom ? `<button data-action="remove-custom-model" data-service-id="${id}" data-model-id="${m.custom}">Remove custom model</button>` : ""}
                    </div>
                </div>`
            );
        }).join("")
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
    const modelCustomEle = document.getElementById("model-custom");
    if (modelCustomEle) {
        modelCustomEle.value = modelCustom;
        modelCustomEle.addEventListener("change", () => {
            modelCustom = modelCustomEle.value;
            render();
        });
    }
}

function showLoadingPage() {
    showHtml(`<div class="loading">Loading...</div>`);
}

function showLoading() {
    const div = document.createElement("div");
    div.innerHTML = `<div class="loading-container"><div class="loading">Loading...</div></div>`;
    document.body.append(div);
    return div
}

root.addEventListener("click", async e => {
    const dataActionEle = e.target.closest("[data-action]");
    if (dataActionEle) {
        const action = dataActionEle.getAttribute("data-action");
        const serviceId = dataActionEle.getAttribute("data-service-id");
        const modelId = dataActionEle.getAttribute("data-model-id");
        if (action === "install-service") {
            const info = await fetchX(`/admin/services/${serviceId}`, {
                method: "GET"
            });
            showInstallServiceModal(info, async data => {
                showLoadingPage();
                try {
                    await fetchX(`/admin/services/${serviceId}`, {
                        method: "POST",
                        body: JSON.stringify({spec: data}),
                        headers: {"Content-Type": "application/json"}
                    });
                }
                catch {}
                showServicesPage();
            });
        }
        else if (action === "uninstall-service") {
            showLoadingPage()
            await fetchX(`/admin/services/${serviceId}`, {
                method: "DELETE",
                body: JSON.stringify({purge: false}),
                headers: {"Content-Type": "application/json"}
            });
            showServicesPage();
        }
        else if (action === "open-service-models") {
            resetModelFilters();
            showServicePage(serviceId);
        }
        else if (action === "install-model") {
            const info = await fetchX(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                method: "GET"
            });
            showInstallModelModal(info, async data => {
                showLoadingPage()
                try {
                    await fetchX(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                        method: "POST",
                        body: JSON.stringify({spec: data}),
                        headers: {"Content-Type": "application/json"}
                    });
                }
                catch {}
                showServicePage(serviceId);
            });
        }
        else if (action === "uninstall-model") {
            showLoadingPage()
            await fetchX(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                method: "DELETE",
                body: JSON.stringify({purge: false}),
                headers: {"Content-Type": "application/json"}
            });
            showServicePage(serviceId);
        }
        else if (action === "add-custom-model") {
            const info = await fetchX(`/admin/services/${serviceId}`, {
                method: "GET"
            });
            showAddCustomModelModal(info, async data => {
                showLoadingPage()
                try {
                    await fetchX(`/admin/services/${serviceId}/models/custom`, {
                        method: "POST",
                        body: JSON.stringify({spec: data}),
                        headers: {"Content-Type": "application/json"}
                    });
                }
                catch {}
                resetModelFilters()
                modelFilter = data.id;
                showServicePage(serviceId);
            });
        }
        else if (action === "remove-custom-model") {
            showLoadingPage()
            await fetchX(`/admin/services/${serviceId}/models/custom/${modelId}`, {
                method: "DELETE",
                body: JSON.stringify({purge: false}),
                headers: {"Content-Type": "application/json"}
            });
            showServicePage(serviceId);
        }
        else if (action === "open-services") {
            showServicesPage();
        }
        else if (action === "show-docker-logs") {
            const loading = showLoading();
            try {
                const result = await fetchX(`/admin/services/${serviceId}/docker/logs${modelId ? `?model_id=${modelId}` : ""}`);
                showContentModal({title: "Docker logs", text: result.logs, wide: true});
            }
            finally {
                loading.remove();
            }
        }
        else if (action === "show-docker-compose-file") {
            const loading = showLoading();
            try {
                const result = await fetchX(`/admin/services/${serviceId}/docker/compose${modelId ? `?model_id=${modelId}` : ""}`);
                showContentModal({title: "Docker compose file", text: result.compose_file, wide: true});
            }
            finally {
                loading.remove();
            }
        }
        else if (action === "restart-docker") {
            const loading = showLoading();
            try {
                await fetchX(`/admin/services/${serviceId}/docker/restart${modelId ? `?model_id=${modelId}` : ""}`, {
                    method: "POST"
                });
                showContentModal({title: "Docker restart", text: "Docker restarted!", pre: false});
            }
            finally {
                loading.remove();
            }
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
    const content = await response.text();
    if (response.status !== 200) {
        console.log("Invalid status code", response.status, content);
        alert(`ERROR: Invalid status code ${response.status} - ${content}`);
        throw new Error("Invalid status code");
    }
    try {
        return JSON.parse(content);
    }
    catch (e) {
        console.log("JSON parse error", content, e);
        alert(`ERROR: Invalid response, cannot parse JSON - ${content}`)
        throw e;
    }
}

function showAdminApiKeyModal() {
    const api_key = prompt("Type admin api key");
    if (api_key) {
        localStorage.setItem(LOCAL_STORAGE_KEY_ADMIN_API_KEY, api_key);
    }
}

const admin_api_key = localStorage.getItem(LOCAL_STORAGE_KEY_ADMIN_API_KEY);
showServicesPage()
