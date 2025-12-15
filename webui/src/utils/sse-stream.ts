/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/

export interface ProgressEvent {
  type: "progress" | "finish";
  stage?: "install" | "download";
  value?: number; // 0.0 to 1.0
  status?: "ok" | "error";
  details?: string;
}

class SSEStream {
  private text = "";

  constructor(private onProgress: (data: ProgressEvent) => void) {}

  consume(chunk: string) {
    this.text += chunk;
    while (true) {
      const index = this.text.indexOf("\n\n");
      if (index === -1) {
        return;
      }

      const event = this.text.substring(0, index);
      this.text = this.text.substring(index + 2);

      if (event.startsWith("data: ")) {
        const data = event.substring(6);
        try {
          const progress: ProgressEvent = JSON.parse(data);
          try {
            this.onProgress(progress);
          } catch (e) {
            console.error("Error during onProgress callback", progress, e);
          }
        } catch (e) {
          console.error("JSON parse error during SSE stream", data, e);
        }
      } else {
        console.log("Unexpected chunk", event);
      }
    }
  }
}

export async function readSSEStream(
  response: Response,
  onProgress: (data: ProgressEvent) => void
): Promise<void> {
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Response body is not readable");
  }

  const decoder = new TextDecoder();
  const stream = new SSEStream(onProgress);

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (value) {
        const chunk = decoder.decode(value, { stream: true });
        stream.consume(chunk);
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}

export function getStageLabel(stage: string): string {
  switch (stage) {
    case "install":
      return "Installing";
    case "download":
      return "Downloading";
    default:
      return stage;
  }
}
