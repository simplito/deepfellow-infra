/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
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
    const data = await fetchJson("/admin/services");
    const html = `
        <div class="page services-page">
            <h3>
                Services
                <button data-action="show-mesh-info">Show mesh info</button>
            </h3>
            <div class="boxes services">
                ${data.list.map(s => {
                    const values = s.installed && !s.installed.stage ? s.installed : false;
                    return (
                        `<div class="box service" data-service-id="${s.id}">
                            <div class="box-name service-name">
                                ${s.id}
                                <div class="badge ${s.installed ? "badge-installed" : "badge-not-installed"}">
                                    ${s.installed ? s.installed.stage ? `${getStageLabel(s.installed.stage)} ${(s.installed.value * 100).toFixed(1)}%` : "Installed" : "Not installed"}
                                </div>
                                ${!s.installed && s.downloaded ? `<div class="badge badge-downloaded">Downloaded</div>` : ""}
                            </div>
                            <div class="box-size service-description">
                                ${typeof(s.description) == "string" && s.description.length > 0 ?
                                    `${s.description}` : ``
                                }
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
                                ${s.installed && !s.installed.stage ? `<button data-action="open-service-models" data-service-id="${s.id}">Models</button>` : ""}
                                ${s.installed && !s.installed.stage ? `<button data-action="uninstall-service" data-service-id="${s.id}">Uninstall</button>` : ""}
                                ${!s.installed && s.downloaded ? `<button data-action="purge-service" data-service-id="${s.id}">Purge</button>` : ""}
                            </div>
                        </div>`
                    );
                }).join("")}
            </div>
        </div>`;
    showHtml(html);
    Promise.all(data.list.map(async service => {
        if (service.installed && service.installed.stage) {
            const response = await fetchForResponse(`/admin/services/${service.id}/progress`);
            readServiceInstallProgress(service.id, response);
        }
    }))
}

let modelFilter = "";
let modelType = "__all";
let modelInstalled = "__all";
let modelDownloaded = "__all";
let modelCustom = "__all"

function resetModelFilters() {
    modelFilter = "";
    modelType = "__all";
    modelInstalled = "__all";
    modelDownloaded = "__all";
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
                        if (field.type === "textarea") {
                            return `<textarea name="${field.name}" ${field.required ? `required="required"` : ""} placeholder="${field.placeholder || ""}">${field.default || ""}</textarea>`;
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
        for (const field of div.querySelectorAll("input,select,textarea")) {
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

function showUninstallModal(options) {
    const html = `
<div class="modal-backdrop">
    <div class="modal">
        <div class="modal-title">${options.title}</div>
        <div class="content"></div>
        <div class="buttons">
            <button data-id="uninstall">Uninstall</button>
            <button data-id="purge">Purge</button>
            <button data-id="cancel">Cancel</button>
        </div>
    </div>
</div>`;
    const div = document.createElement("div");
    div.innerHTML = html;
    div.querySelector(".content").textContent = options.text;
    document.body.append(div);
    div.querySelector("[data-id=uninstall]").addEventListener("click", () => {
        options.onResult("uninstall")
        div.remove();
    });
    div.querySelector("[data-id=purge]").addEventListener("click", () => {
        options.onResult("purge")
        div.remove();
    });
    div.querySelector("[data-id=cancel]").addEventListener("click", () => {
        div.remove();
    });
}


function showTestResultModal(options) {
    const html = `
<div class="modal-backdrop">
    <div class="modal ${options.wide ? "modal-wide" : ""}">
        <div class="modal-title">${options.title}</div>
        <div style="margin: 0 15px;">
            ${options.result.error ? `<div class="text-result-error"><span>x</span> Test failed!</div>` : ""}
            ${options.result.result ? `<div class="text-result-success"><span>✓</span> Test passed!</div>` : ""}
            ${options.result.output ? `<h3>Output:</h3><div class="output"></div>` : ""}
            ${options.result.details ? `<h3>Details:</h3><pre class="content" style="max-height: 150px;"></pre>` : ""}
        </div>
        <div class="buttons">
            <button data-id="ok">OK</button>
        </div>
    </div>
</div>`;
    const div = document.createElement("div");
    div.innerHTML = html;
    if (options.result.details) {
        div.querySelector(".content").textContent = JSON.stringify(options.result.details, null, 2);
    }
    const outputEle = div.querySelector(".output");
    if (options.result.output) {
        if (typeof(options.result.output) === "object") {
            if (options.result.output.content_type.startsWith("audio/")) {
                const audio = document.createElement("audio");
                audio.setAttribute("controls", "controls");
                const source = document.createElement("source");
                source.src = `data:${options.result.output.content_type};base64,${options.result.output.data}`;
                audio.append(source);
                outputEle.append(audio);
            }
            else if (options.result.output.content_type.startsWith("image/")) {
                const img = document.createElement("img");
                img.style = "max-width: 100px; max-height: 100px;";
                img.src = `data:${options.result.output.content_type};base64,${options.result.output.data}`;
                outputEle.append(img);
            }
            else {
                outputEle.textContent = `Content with type ${options.result.output.content_type}`;
            }
        }
        else {
            outputEle.textContent = options.result.output
        }
    }
    document.body.append(div);
    div.querySelector("[data-id=ok]").addEventListener("click", () => {
        div.remove();
    });
}

function showConfirmAlert(options) {
    return new Promise((resolve) => {
        const html = `
            <div class="modal-backdrop">
                <div class="modal">
                    <div class="modal-title">${options.title}</div>
                    <div class="content"></div>
                    <div class="buttons">
                        <button data-id="continue">Continue</button>
                        <button data-id="cancel">Cancel</button>
                    </div>
                </div>
            </div>`;
            const div = document.createElement("div");
            div.innerHTML = html;
            div.querySelector(".content").textContent = options.text;
            document.body.append(div);
            div.querySelector("[data-id=continue]").addEventListener("click", () => {
                div.remove();
                resolve(true);
            });
            div.querySelector("[data-id=cancel]").addEventListener("click", () => {
                div.remove();
                resolve(false);
            });
    });
}

async function showServicePage(id) {
    showLoadingPage();
    const serivceInfo = await fetchJson(`/admin/services/${id}`);
    const data = await fetchJson(`/admin/services/${id}/models`);
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
    const downloadedOptions = {
        __all: null,
        downloaded: true,
        notdownloaded: false
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
                
                <span style="margin-left: 20px;">Downloaded:</span>
                <select id="model-downloaded">
                    <option value="__all">--ALL--</option>
                    <option value="downloaded">Only downloaded</option>
                    <option value="notdownloaded">Only not downloaded</option>
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
        const downloaded = downloadedOptions[modelDownloaded];
        const custom = customOptions[modelCustom];
        const filtered = list.filter(x =>
            (!filterl || x.id.toLowerCase().includes(filterl)) &&
            (modelType === "__all" || x.type === modelType) &&
            (installed === null || x.installed === installed) &&
            (downloaded === null || x.downloaded === downloaded) &&
            (custom === null || (!!x.custom) === custom)
        );
        boxes.innerHTML = filtered.length === 0 ? `<div class="empty">No elements</div>` : filtered.map(m => {
            const values = m.installed ? m.installed.spec || {} : false;
            return (
                `<div class="box model" data-model-id="${m.id}">
                    <div class="box-name model-name">
                        ${m.id}
                        <div class="badge ${m.installed ? "badge-installed" : "badge-not-installed"}">
                            ${m.installed ? m.installed.stage ? `${getStageLabel(m.installed.stage)} ${(m.installed.value * 100).toFixed(1)}%` : "Installed" : "Not installed"}
                        </div>
                        ${!m.installed && m.downloaded ? `<div class="badge badge-downloaded">Downloaded</div>` : ""}
                        ${m.custom ? `<div class="badge">
                            Custom
                        </div>` : ""}
                    </div>
                    <div class="model-type">${types[m.type] || m.type}</div>
                    <div class="box-size model-size">
                        Size: <span class="size">${m.size ? m.size : "N/A"}</span>
                    </div>
                    <div class="box-values">
                        ${values ? Object.keys(values).map(key => `<div class="box-value">${key} = ${m.spec.fields.find(x => x.name === key)?.type === "password" ? "*****" : (values[key] === null ? "" : values[key])}</div>`).join("") : ""}
                    </div>
                    <div class="box-buttons model-buttons">
                        ${!m.installed ? `<button data-action="install-model" data-service-id="${id}" data-model-id="${m.id}">Install</button>` : ""}
                        ${!m.installed && m.downloaded ? `<button data-action="purge-model" data-service-id="${id}" data-model-id="${m.id}">Purge</button>` : ""}
                        ${m.installed && !m.installed.stage ? `<button data-action="uninstall-model" data-service-id="${id}" data-model-id="${m.id}">Uninstall</button>` : ""}
                        ${m.installed && !m.installed.stage ? `<button data-action="test-model" data-service-id="${id}" data-model-id="${m.id}">Test</button>` : ""}
                        ${m.installed && !m.installed.stage && m.has_docker ? `
                            <button data-service-id="${id}" data-model-id="${m.id}" data-action="show-docker-logs">Show docker logs</button>
                            <button data-service-id="${id}" data-model-id="${m.id}" data-action="show-docker-compose-file">Show docker compose file</button>
                            <button data-service-id="${id}" data-model-id="${m.id}" data-action="restart-docker">Restart docker</button>` : ""}
                        ${!m.installed && m.custom ? `<button data-action="remove-custom-model" data-service-id="${id}" data-model-id="${m.custom}">Remove custom model</button>` : ""}
                    </div>
                </div>`
            );
        }).join("")
        Promise.all(filtered.map(async model => {
            if (model.installed && model.installed.stage) {
                const response = await fetchForResponse(`/admin/services/${id}/models/progress?model_id=${model.id}`);
                readModalInstallProgress(id, model.id, response);
            }
        }));
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
    const modelDownloadedEle = document.getElementById("model-downloaded");
    modelDownloadedEle.value = modelDownloaded;
    modelDownloadedEle.addEventListener("change", () => {
        modelDownloaded = modelDownloadedEle.value;
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
    showHtml(
        `<div class="loading">
            <div>Loading...</div>
            <div id="progress"></div>
        </div>`);
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
            const info = await fetchJson(`/admin/services/${serviceId}`, {
                method: "GET"
            });
            showInstallServiceModal(info, async data => {
                const response = await fetchForResponse(`/admin/services/${serviceId}`, {
                    method: "POST",
                    body: JSON.stringify({stream: true, spec: data}),
                    headers: {"Content-Type": "application/json"},
                    onError: async (response, content) => {
                        if (response.status === 400) {
                            try {
                                const errorData = JSON.parse(content);
                                if (!errorData || !errorData.detail || !errorData.detail.warnings) {
                                    return false;
                                }
                                const res = await showConfirmAlert({
                                    title: "Install " + serviceId,
                                    text: `There are warnings, do you really want to continue: ${errorData.detail.warnings.join(",")}`
                                });
                                if (res) {
                                    return fetchForResponse(`/admin/services/${serviceId}`, {
                                        method: "POST",
                                        body: JSON.stringify({stream: true, spec: data, ignore_warnings: true}),
                                        headers: {"Content-Type": "application/json"},
                                    });
                                }
                                else {
                                    return true;
                                }
                            }
                            catch (e) {
                                console.log("Second try error", e);
                            }
                        }
                        return false;
                    },
                });
                readServiceInstallProgress(serviceId, response);
            });
        }
        else if (action === "uninstall-service") {
            showUninstallModal({
                title: "Uninstalling Model",
                text: "How do you want to uninstall service? It purge all service files.",
                onResult: async (removeType) => {
                    showLoadingPage()
                    await fetchJson(`/admin/services/${serviceId}`, {
                        method: "DELETE",
                        body: JSON.stringify({purge: removeType === "purge"}),
                        headers: {"Content-Type": "application/json"}
                    });
                    showServicesPage();
                }
            })
        }
        else if (action === "purge-service") {
            showLoadingPage()
            await fetchJson(`/admin/services/${serviceId}`, {
                method: "DELETE",
                body: JSON.stringify({purge: true}),
                headers: {"Content-Type": "application/json"}
            });
            showServicesPage();
        }
        else if (action === "open-service-models") {
            resetModelFilters();
            showServicePage(serviceId);
        }
        else if (action === "install-model") {
            const info = await fetchJson(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                method: "GET"
            });
            showInstallModelModal(info, async data => {
                const response = await fetchForResponse(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                    method: "POST",
                    body: JSON.stringify({stream: true, spec: data}),
                    headers: {"Content-Type": "application/json"},
                    onError: async (response, content) => {
                        if (response.status === 400) {
                            try {
                                const errorData = JSON.parse(content);
                                if (!errorData || !errorData.detail || !errorData.detail.warnings) {
                                    return false;
                                }
                                const res = await showConfirmAlert({
                                    title: "Install " + modelId,
                                    text: `There are warnings, do you really want to continue: ${errorData.detail.warnings.join(",")}`
                                });
                                if (res) {
                                    return fetchForResponse(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                                        method: "POST",
                                        body: JSON.stringify({stream: true, spec: data, ignore_warnings: true}),
                                        headers: {"Content-Type": "application/json"},
                                    });
                                }
                                else {
                                    return true;
                                }
                            }
                            catch (e) {
                                console.log("Second try error", e);
                            }
                        }
                        return false;
                    },
                });
                readModalInstallProgress(serviceId, modelId, response);
            });
        }
        else if (action === "uninstall-model") {
            showUninstallModal({
                title: "Uninstalling Model",
                text: "How do you want to uninstall model? Purge remove all model files.",
                onResult: async (removeType) => {
                    showLoadingPage()
                    await fetchJson(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                        method: "DELETE",
                        body: JSON.stringify({purge: removeType === "purge"}),
                        headers: {"Content-Type": "application/json"}
                    });
                    showServicePage(serviceId);
                }
            })
        }
        else if (action === "purge-model") {
            showLoadingPage()
            await fetchJson(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                method: "DELETE",
                body: JSON.stringify({purge: true}),
                headers: {"Content-Type": "application/json"}
            });
            showServicePage(serviceId);
        }
        else if (action === "test-model") {
            const loading = showLoading();
            try {
                const info = await fetchJson(`/admin/services/${serviceId}/models/_?model_id=${encodeURIComponent(modelId)}`, {
                    method: "GET"
                });
                if (!info.installed) {
                    throw new Error("Model not installed");
                }
                const result = await fetchJson(
                    `/admin/services/model/test/${info.installed.registration_id}`,
                );
                loading.remove();
                showTestResultModal({
                    title: `Test: ${modelId}`,
                    result: result
                });
            }
            catch (error) {
                console.log("Error", error);
                loading.remove();
            }
        }
        else if (action === "add-custom-model") {
            const info = await fetchJson(`/admin/services/${serviceId}`, {
                method: "GET"
            });
            showAddCustomModelModal(info, async data => {
                showLoadingPage()
                try {
                    await fetchJson(`/admin/services/${serviceId}/models/custom`, {
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
            await fetchJson(`/admin/services/${serviceId}/models/custom/${modelId}`, {
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
                const result = await fetchJson(`/admin/services/${serviceId}/docker/logs${modelId ? `?model_id=${modelId}` : ""}`);
                showContentModal({title: "Docker logs", text: result.logs, wide: true});
            }
            finally {
                loading.remove();
            }
        }
        else if (action === "show-docker-compose-file") {
            const loading = showLoading();
            try {
                const result = await fetchJson(`/admin/services/${serviceId}/docker/compose${modelId ? `?model_id=${modelId}` : ""}`);
                showContentModal({title: "Docker compose file", text: result.compose_file, wide: true});
            }
            finally {
                loading.remove();
            }
        }
        else if (action === "restart-docker") {
            const loading = showLoading();
            try {
                await fetchJson(`/admin/services/${serviceId}/docker/restart${modelId ? `?model_id=${modelId}` : ""}`, {
                    method: "POST"
                });
                showContentModal({title: "Docker restart", text: "Docker restarted!", pre: false});
            }
            finally {
                loading.remove();
            }
        }
        else if (action === "show-mesh-info") {
            const loading = showLoading();
            try {
                const result = await fetchJson("/admin/mesh/info");
                showContentModal({title: "Mesh info", text: JSON.stringify(result, null, 2)});
            }
            finally {
                loading.remove();
            }
        }
    }
});

const LOCAL_STORAGE_KEY_ADMIN_API_KEY = "admin_api_key";

async function fetchForResponse(url, options) {
    const opts = options || {};
    opts.headers = opts.headers || {};
    opts.headers["Authorization"] = "Bearer " + admin_api_key;
    const response = admin_api_key ? await fetch(url, opts) : {status: 403};
    if (response.status === 401 || response.status === 403) {
        showAdminApiKeyModal();
        document.location.reload();
        throw new Error("Unauthorized");
    }
    if (response.status !== 200) {
        const content = await response.text();
        const handler = opts.onError ? await opts.onError(response, content) : false;
        if (handler !== false && handler !== true) {
            return handler;
        }
        console.log("Invalid status code", response.status, content);
        if (handler !== true) {
            alert(`ERROR: Invalid status code ${response.status} - ${content}`);
        }
        throw new Error("Invalid status code");
    }
    return response;
}

async function fetchJson(url, options) {
    const response = await fetchForResponse(url, options);
    const content = await response.text();
    try {
        return JSON.parse(content);
    }
    catch (e) {
        console.log("JSON parse error", content, e);
        alert(`ERROR: Invalid response, cannot parse JSON - ${content}`)
        throw e;
    }
}

function getStageLabel(stage) {
    if (stage == "install") {
        return "Installing";
    }
    if (stage == "download") {
        return "Downloading";
    }
    return stage || "";
}

function readProgress(response, onFinish) {
    const progress = document.getElementById("progress");
    readProgressCore(response, onFinish, data => {
        // console.log(data);
        if (!data) {
            return;
        }
        if (data.type === "progress") {
            if (progress) progress.textContent = getStageLabel(data.stage) + " " + (Math.floor(data.value * 10000) / 100).toFixed(1) + "%";
        }
        else if (data.type === "finish") {
            if (data.status === "ok") {
                if (progress) progress.textContent = data.details;
            }
            if (data.status === "error") {
                console.log("Error", data.details);
                alert("Error: " + data.details);
            }
        }
    });
}

function readServiceInstallProgress(serviceId, response) {
    const serviceBox = document.querySelector(`.box.service[data-service-id="${serviceId}"]`);
    const badge = serviceBox ? serviceBox.querySelector(".badge") : null;
    if (badge) {
        badge.classList.add("badge-installed");
        badge.textContent = "...";
    }
    if (serviceBox) {
        const buttons = serviceBox.querySelector(".box-buttons");
        if (buttons) {
            buttons.style.display = "none";
        }
    }
    readProgressCore(response, () => {
        showServicesPage();
    }, data => {
        if (!data) {
            return;
        }
        if (data.type === "progress") {
            if (badge) badge.textContent = `${getStageLabel(data.stage)} ${(data.value * 100).toFixed(1)}%`
        }
        else if (data.type === "finish") {
            if (data.status === "ok") {
                // do nothing, onFinish call showServicesPage
            }
            if (data.status === "error") {
                console.log("Error", data.details);
                alert("Error: " + data.details);
            }
        }
    });
}

function readModalInstallProgress(serviceId, modelId, response) {
    const modelBox = document.querySelector(`.box.model[data-model-id="${modelId}"]`);
    const badge = modelBox ? modelBox.querySelector(".badge") : null;
    if (badge) {
        badge.classList.add("badge-installed");
        badge.textContent = "...";
    }
    if (modelBox) {
        const buttons = modelBox.querySelector(".box-buttons");
        if (buttons) {
            buttons.style.display = "none";
        }
    }
    readProgressCore(response, () => {
        showServicePage(serviceId);
    }, data => {
        if (!data) {
            return;
        }
        if (data.type === "progress") {
            if (badge) badge.textContent = `${getStageLabel(data.stage)} ${(data.value * 100).toFixed(1)}%`
        }
        else if (data.type === "finish") {
            if (data.status === "ok") {
                // do nothing, onFinish call showServicePage
            }
            if (data.status === "error") {
                console.log("Error", data.details);
                alert("Error: " + data.details);
            }
        }
    });
}

function readProgressCore(response, onFinish, onChunk) {
    const contentType = (response.headers.get("content-type") || "").split(";")[0].trim();
    if (contentType !== "text/event-stream") {
        (async () => {
            const content = await response.text();
            try {
                return JSON.parse(content);
            }
            catch (e) {
                console.log("JSON parse error", content, e);
                alert(`ERROR: Invalid response, cannot parse JSON - ${content}`)
                throw e;
            }
            finally {
                onFinish();
            }
        })()
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const stream = new SSEStream(onChunk);

    function readChunk() {
        return reader.read().then(({ done, value }) => {
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                stream.consume(chunk);
            }
            if (done) {
                onFinish();
                return;
            }
            return readChunk();
        });
    }
    readChunk();
}

class SSEStream {
    
    constructor(onProgress) {
        this.text = "";
        this.onProgress = onProgress;
    }
    
    consume(txt) {
        this.text += txt;
        while (true) {
            const index = this.text.indexOf("\n\n");
            if (index === -1) {
                return;
            }
            const chunk = this.text.substring(0, index);
            this.text = this.text.substring(index + 2);
            if (chunk.startsWith("data: ")) {
                const data = chunk.substring(6);
                try {
                    const progress = JSON.parse(data);
                    try {
                        this.onProgress(progress);
                    }
                    catch (e) {
                        console.log("Error during onProgress callback", progress, e);
                    }
                }
                catch (e) {
                    console.log("JSON parse error during reading streaming chunk", data);
                }
            }
            else {
                console.log("Unexpected chunk", chunk);
            }
        }
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
