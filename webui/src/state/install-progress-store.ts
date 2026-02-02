import { useSyncExternalStore } from "react";

export type InstallStage = "install" | "download";

export type InstallProgress = {
	stage: InstallStage;
	value: number;
};

const STAGE_ORDER: Record<InstallStage, number> = {
	download: 0,
	install: 1,
};

type Snapshot = {
	services: Record<string, InstallProgress | undefined>;
	models: Record<string, InstallProgress | undefined>;
};

let snapshot: Snapshot = {
	services: {},
	models: {},
};

const listeners = new Set<() => void>();

function emit() {
	for (const listener of listeners) listener();
}

export function subscribe(listener: () => void) {
	listeners.add(listener);
	return () => {
		listeners.delete(listener);
	};
}

export function getSnapshot(): Snapshot {
	return snapshot;
}

function setSnapshot(next: Snapshot) {
	snapshot = next;
	emit();
}

function normalizeProgress(progress: InstallProgress | undefined): InstallProgress | undefined {
	if (!progress) return undefined;
	const value = progress.value;
	if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
	// Value is expected to be 0..1; clamp to avoid UI flicker on bad input.
	const clamped = Math.min(1, Math.max(0, value));
	if (clamped === value) return progress;
	return { ...progress, value: clamped };
}

function shouldIgnoreProgressUpdate(prev: InstallProgress | undefined, next: InstallProgress | undefined): boolean {
	if (!prev || !next) return false;

	// Keep stage monotonic (download -> install). This avoids UI getting stuck
	// when a secondary progress stream reports an earlier stage after we already
	// advanced to a later one.
	const prevStageOrder = STAGE_ORDER[prev.stage] ?? 0;
	const nextStageOrder = STAGE_ORDER[next.stage] ?? 0;
	if (nextStageOrder < prevStageOrder) return true;

	// If the stage is unchanged, keep progress monotonic to avoid brief regressions
	// when reconnecting to streaming endpoints.
	if (prev.stage === next.stage && next.value < prev.value) return true;
	return false;
}

function modelKey(serviceId: string, modelId: string) {
	return `${serviceId}::${modelId}`;
}

export function setServiceInstallProgress(serviceId: string, progress: InstallProgress | undefined) {
	const next = normalizeProgress(progress);
	const prev = snapshot.services[serviceId];
	if (shouldIgnoreProgressUpdate(prev, next)) return;
	if (prev?.stage === next?.stage && prev?.value === next?.value) return;
	setSnapshot({
		...snapshot,
		services: {
			...snapshot.services,
			[serviceId]: next,
		},
	});
}

export function clearServiceInstallProgress(serviceId: string) {
	const next = { ...snapshot.services };
	delete next[serviceId];
	setSnapshot({
		...snapshot,
		services: next,
	});
}

export function setModelInstallProgress(serviceId: string, modelId: string, progress: InstallProgress | undefined) {
	const key = modelKey(serviceId, modelId);
	const next = normalizeProgress(progress);
	const prev = snapshot.models[key];
	if (prev?.stage === next?.stage && prev?.value === next?.value) return;
	setSnapshot({
		...snapshot,
		models: {
			...snapshot.models,
			[key]: next,
		},
	});
}

export function clearModelInstallProgress(serviceId: string, modelId: string) {
	const key = modelKey(serviceId, modelId);
	const next = { ...snapshot.models };
	delete next[key];
	setSnapshot({
		...snapshot,
		models: next,
	});
}

export function useInstallProgressSnapshot(): Snapshot {
	return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function useModelInstallProgress(serviceId: string, modelId: string): InstallProgress | undefined {
	const key = modelKey(serviceId, modelId);
	return useSyncExternalStore(
		subscribe,
		() => getSnapshot().models[key],
		() => getSnapshot().models[key]
	);
}
