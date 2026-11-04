// @vitest-environment jsdom
/*
DeepFellow Software Framework.
Copyright © 2026 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const { captureException } = vi.hoisted(() => ({
  captureException: vi.fn(),
}));

vi.mock("@sentry/react", () => ({ captureException }));

import { RootErrorFallback } from "./RootErrorFallback";

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("RootErrorFallback", () => {
  it("shows a fallback screen and reports the error to Sentry", () => {
    const error = new Error("render exploded");

    render(<RootErrorFallback error={error} reset={vi.fn()} />);

    expect(screen.getByText("Something went wrong")).toBeTruthy();
    expect(screen.getByText("render exploded")).toBeTruthy();
    expect(captureException).toHaveBeenCalledTimes(1);
    expect(captureException).toHaveBeenCalledWith(error);
  });

  it("resets the error boundary on Try again", () => {
    const reset = vi.fn();

    render(<RootErrorFallback error={new Error("x")} reset={reset} />);
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(reset).toHaveBeenCalledTimes(1);
  });
});
