/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { LogOut, User } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  SidebarMenu,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

interface NavUserProps {
  user: {
    name: string;
  };
  onLogout: () => void;
}

export function NavUser({ user, onLogout }: NavUserProps) {
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <div className="flex items-center gap-3 p-2">
          <Avatar className="size-8 rounded-lg">
            <AvatarFallback className="rounded-lg bg-primary text-primary-foreground">
              <User className="size-4" />
            </AvatarFallback>
          </Avatar>
          <div className="flex-1 min-w-0 text-left text-sm leading-tight">
            <span className="truncate font-semibold">{user.name}</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={onLogout}
            className="size-8 shrink-0"
            title="Log out"
          >
            <LogOut className="size-4" />
            <span className="sr-only">Log out</span>
          </Button>
        </div>
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
