/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

import {
	Breadcrumb,
	BreadcrumbItem,
	BreadcrumbLink,
	BreadcrumbList,
	BreadcrumbPage,
	BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Separator } from "@/components/ui/separator";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Link } from "@tanstack/react-router";

interface SiteHeaderProps {
	breadcrumbs?: Array<{
		label: string;
		href?: string;
	}>;
}

export function SiteHeader({ breadcrumbs }: SiteHeaderProps) {
	const isExternalHref = (href: string) =>
		href.startsWith("http://") ||
		href.startsWith("https://") ||
		href.startsWith("mailto:") ||
		href.startsWith("tel:");

	return (
		<header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
			<SidebarTrigger className="-ml-1" />
			<Separator orientation="vertical" className="mr-2 h-4" />
			{breadcrumbs && breadcrumbs.length > 0 && (
				<Breadcrumb>
					<BreadcrumbList>
						{breadcrumbs.map((crumb, index) => (
							<div
								key={crumb.href ?? crumb.label}
								className="flex items-center gap-2"
							>
								{index > 0 && <BreadcrumbSeparator />}
								<BreadcrumbItem>
									{crumb.href ? (
										isExternalHref(crumb.href) ? (
											<BreadcrumbLink
												href={crumb.href}
												target="_blank"
												rel="noopener noreferrer"
											>
												{crumb.label}
											</BreadcrumbLink>
										) : (
											<BreadcrumbLink asChild>
												<Link to={crumb.href}>{crumb.label}</Link>
											</BreadcrumbLink>
										)
									) : (
										<BreadcrumbPage>{crumb.label}</BreadcrumbPage>
									)}
								</BreadcrumbItem>
							</div>
						))}
					</BreadcrumbList>
				</Breadcrumb>
			)}
		</header>
	);
}
