import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Skeleton } from "@/components/ui/skeleton";
import type { SpecField } from "@/deepfellow/types";
/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { useCallback, useEffect, useMemo, useState } from "react";
import { ListInput } from "./ListInput";
import { MapInput } from "./MapInput";

interface DynamicFormModalProps {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	title: string;
	fields: SpecField[];
	onSubmit: (data: Record<string, unknown>) => void;
	isSubmitting?: boolean;
	isLoading?: boolean;
	deferRender?: boolean;
	submitLabel?: string;
	submittingLabel?: string;
}

export function DynamicFormModal({
	open,
	onOpenChange,
	title,
	fields,
	onSubmit,
	isSubmitting = false,
	isLoading = false,
	deferRender = false,
	submitLabel = "Install",
	submittingLabel = "Installing...",
}: DynamicFormModalProps) {
	const [renderFields, setRenderFields] = useState<SpecField[]>(
		deferRender ? [] : fields,
	);

	useEffect(() => {
		if (!open) return;
		if (!deferRender) {
			setRenderFields(fields);
			return;
		}

		// Let the modal paint first (skeleton), then render the full form.
		setRenderFields([]);
		const timeoutId = window.setTimeout(() => setRenderFields(fields), 0);
		return () => window.clearTimeout(timeoutId);
	}, [open, deferRender, fields]);

	const effectiveIsLoading =
		isLoading || (deferRender && open && renderFields.length === 0 && fields.length > 0);

	const initializeFormData = useMemo(() => {
		const initial: Record<string, unknown> = {};
		for (const field of renderFields) {
			if (field.type === "bool") {
				initial[field.name] = field.default === true || field.default === "true";
			} else if (field.type === "list") {
				if (typeof field.default === 'string' && field.default.startsWith('[')) {
					try {
						initial[field.name] = JSON.parse(field.default) ?? [];
					} catch (e) {
						initial[field.name] = [];
					}
				} else {
					initial[field.name] = Array.isArray(field.default) ? field.default : [];
				}
			} else if (field.type === "map") {
				if (typeof field.default === 'string' && field.default.startsWith('{')) {
					try {
						initial[field.name] = JSON.parse(field.default) ?? {};
					} catch (e) {
						initial[field.name] = {};
					}
				} else {
					initial[field.name] = field.default ?? {};
				}
			} else if (field.default !== undefined && field.default !== null) {
				initial[field.name] = field.default;
			}
		}
		return initial;
	}, [renderFields]);

	const [formData, setFormData] =
		useState<Record<string, unknown>>(initializeFormData);
	const [errors, setErrors] = useState<Record<string, string>>({});

	// Reset form data when modal opens or fields change
	useEffect(() => {
		if (open) {
			setFormData(initializeFormData);
			setErrors({});
		}
	}, [open, initializeFormData]);

	// Conditional field visibility logic
	const isFieldVisible = useCallback(
		(field: SpecField): boolean => {
			if (!field.display) return true;
			const [name, value] = field.display.split("=");
			if (name && value) {
				return formData[name] === value;
			}
			return true;
		},
		[formData],
	);

	// Filter fields based on visibility
	const visibleFields = useMemo(() => {
		return renderFields.filter(isFieldVisible);
	}, [renderFields, isFieldVisible]);

	const focusField = (name: string) => {
		const container = document.querySelector(`[data-field-name="${CSS.escape(name)}"]`) as HTMLElement | null;
		if (!container) return;
		const focusable = container.querySelector<HTMLElement>(
			"input,button,textarea,select,[role='combobox'],[role='button']",
		);
		focusable?.focus();
	};

	const validate = (): Record<string, string> => {
		const nextErrors: Record<string, string> = {};

		for (const field of visibleFields) {
			if (!field.required) continue;

			const value = formData[field.name];
			switch (field.type) {
				case "oneof": {
					if (typeof value !== "string" || !value.trim()) {
						nextErrors[field.name] = "This field is required.";
					}
					break;
				}
				case "list": {
					const list = Array.isArray(value) ? value : [];
					const nonEmpty = list.filter((v) => typeof v === "string" && v.trim().length > 0);
					if (nonEmpty.length === 0) {
						nextErrors[field.name] = "Please add at least one item.";
					}
					break;
				}
				case "map": {
					const map = value && typeof value === "object" ? (value as Record<string, unknown>) : {};
					const keys = Object.keys(map).filter((k) => k.trim().length > 0);
					if (keys.length === 0) {
						nextErrors[field.name] = "Please add at least one pair.";
					}
					break;
				}
				case "number": {
					if (typeof value !== "number" || Number.isNaN(value)) {
						nextErrors[field.name] = "This field is required.";
					}
					break;
				}
				case "bool": {
					// Leave as-is: required booleans are not enforced as true.
					break;
				}
				default: {
					if (value === null || value === undefined || String(value).trim().length === 0) {
						nextErrors[field.name] = "This field is required.";
					}
					break;
				}
			}
		}

		return nextErrors;
	};

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();
		if (effectiveIsLoading || isSubmitting) return;
		const nextErrors = validate();
		setErrors(nextErrors);
		const firstInvalid = Object.keys(nextErrors)[0];
		if (firstInvalid) {
			focusField(firstInvalid);
			return;
		}
		onSubmit(formData);
	};

	const handleInputChange = (name: string, value: unknown) => {
		setFormData((prev) => ({ ...prev, [name]: value }));
		setErrors((prev) => {
			if (!prev[name]) return prev;
			const next = { ...prev };
			delete next[name];
			return next;
		});
	};

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
				<DialogContent
					className="sm:max-w-[600px]"
					onInteractOutside={(e) => {
						if (isSubmitting) e.preventDefault();
					}}
					onEscapeKeyDown={(e) => {
						if (isSubmitting) e.preventDefault();
					}}
				>
					<DialogHeader>
						<DialogTitle>{title}</DialogTitle>
						<DialogDescription>
							{effectiveIsLoading ? "Loading form..." : "Fill in the required information below."}
						</DialogDescription>
					</DialogHeader>
					<form onSubmit={handleSubmit}>
						<div className="grid gap-4 px-2 py-4 max-h-[70vh] overflow-auto">
							{effectiveIsLoading ? (
								<div className="grid gap-4">
									<div className="grid gap-2">
										<Skeleton className="h-4 w-40" />
										<Skeleton className="h-10 w-full" />
									</div>
									<div className="grid gap-2">
										<Skeleton className="h-4 w-48" />
										<Skeleton className="h-10 w-full" />
									</div>
									<div className="grid gap-2">
										<Skeleton className="h-4 w-36" />
										<Skeleton className="h-10 w-full" />
									</div>
								</div>
							) : (
								visibleFields.map((field) => (
								<div key={field.name} className="grid gap-2" data-field-name={field.name}>
								<Label htmlFor={field.name}>
									{field.description}
									{field.required && (
										<span className="text-destructive text-sm ml-1">*</span>
									)}
									{!field.required && (
										<span className="text-muted-foreground text-sm ml-1">
											(optional)
										</span>
									)}
								</Label>
								{field.type === "bool" ? (
									<div className="flex items-center space-x-2">
										<Checkbox
											id={field.name}
											checked={formData[field.name] === true}
											onCheckedChange={(checked) =>
												handleInputChange(field.name, checked === true)
											}
										/>
									</div>
								) : field.type === "oneof" ? (
									<Select
										value={(formData[field.name] as string | undefined) || "__none__"}
										onValueChange={(value) =>
											handleInputChange(field.name, value === "__none__" ? "" : value)
										}
									>
										<SelectTrigger id={field.name} className="w-full">
											<SelectValue placeholder="Select an option" />
										</SelectTrigger>
										<SelectContent>
											<SelectItem value="__none__"><span className="text-muted-foreground">None</span></SelectItem>
											{field.values?.filter((val) => (typeof val === "string" ? val : val.value) !== "").map((val) => (
												<SelectItem key={typeof(val) === "string" ? val : val.value} value={typeof(val) === "string" ? val : val.value}>
													{typeof(val) === "string" ? val : val.label}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								) : field.type === "list" ? (
									<ListInput
										value={(formData[field.name] as string[] | undefined) ?? []}
										onChange={(value) => handleInputChange(field.name, value)}
										placeholder={field.placeholder}
									/>
								) : field.type === "map" ? (
									<MapInput
										value={
											(formData[field.name] as
												| Record<string, string>
												| undefined) ?? {}
										}
										onChange={(value) => handleInputChange(field.name, value)}
										placeholder={field.placeholder}
									/>
								) : field.type === "textarea" ? (
									<Textarea
										id={field.name}
										placeholder={field.placeholder || ""}
										required={field.required}
										value={(formData[field.name] as string | undefined) ?? ""}
										onChange={(e) =>
											handleInputChange(field.name, e.target.value)
										}
										aria-invalid={!!errors[field.name]}
									/>
								) : (
									<Input
										id={field.name}
										type={field.type}
										placeholder={field.placeholder || ""}
										required={field.required}
										value={
											(formData[field.name] as string | number | undefined) ??
											""
										}
										onChange={(e) => {
											const value =
												field.type === "number"
													? e.target.valueAsNumber
													: e.target.value;
											handleInputChange(field.name, value);
										}}
										aria-invalid={!!errors[field.name]}
									/>
								)}
								{errors[field.name] && (
									<div className="text-sm text-destructive">{errors[field.name]}</div>
								)}
								</div>
							))
							)}
						</div>
						<DialogFooter>
							<Button
								type="button"
								variant="outline"
								onClick={() => onOpenChange(false)}
								disabled={isSubmitting}
							>
								Cancel
							</Button>
							<Button type="submit" disabled={isSubmitting || effectiveIsLoading}>
								{isSubmitting ? submittingLabel : submitLabel}
							</Button>
						</DialogFooter>
					</form>
			</DialogContent>
		</Dialog>
	);
}
