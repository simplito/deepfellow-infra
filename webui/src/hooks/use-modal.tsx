/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { createContext, useContext, useState, useCallback, type ReactNode, type ComponentType } from "react";

/**
 * Base props that all modal components must accept.
 * The ModalProvider will inject these automatically.
 */
export interface ModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

interface ModalState {
  Component: ComponentType<any>;
  props: Record<string, any>;
}

interface ModalContextValue {
  /**
   * Open a modal with a component and its props.
   * The `open` and `onOpenChange` props are injected automatically.
   * 
   * @example
   * modal.open(ConfirmModal, {
   *   title: "Delete item?",
   *   onConfirm: () => deleteItem(),
   * });
   */
  open<P extends ModalProps>(
    Component: ComponentType<P>,
    props: Omit<P, keyof ModalProps>
  ): { close: () => void };
  
  /**
   * Close the currently open modal.
   */
  close(): void;
}

const ModalContext = createContext<ModalContextValue | null>(null);

interface ModalProviderProps {
  children: ReactNode;
}

export function ModalProvider({ children }: ModalProviderProps) {
  const [modalState, setModalState] = useState<ModalState | null>(null);

  const close = useCallback(() => {
    setModalState(null);
  }, []);

  const open = useCallback(<P extends ModalProps>(
    Component: ComponentType<P>,
    props: Omit<P, keyof ModalProps>
  ) => {
    setModalState({ Component, props });
    return { close };
  }, [close]);

  const handleOpenChange = useCallback((isOpen: boolean) => {
    if (!isOpen) {
      close();
    }
  }, [close]);

  const contextValue: ModalContextValue = {
    open,
    close,
  };

  return (
    <ModalContext.Provider value={contextValue}>
      {children}
      {modalState && (
        <modalState.Component
          {...modalState.props}
          open={true}
          onOpenChange={handleOpenChange}
        />
      )}
    </ModalContext.Provider>
  );
}

/**
 * Hook to access the modal system.
 * 
 * @example
 * const modal = useModal();
 * 
 * // Open a confirm modal
 * modal.open(ConfirmModal, {
 *   title: "Delete?",
 *   description: "This cannot be undone.",
 *   onConfirm: () => deleteMutation.mutate(),
 * });
 * 
 * // Open a content modal
 * modal.open(ContentModal, {
 *   title: "Logs",
 *   content: logData,
 *   wide: true,
 * });
 */
export function useModal(): ModalContextValue {
  const context = useContext(ModalContext);
  if (!context) {
    throw new Error("useModal must be used within a ModalProvider");
  }
  return context;
}

