/*
DeepFellow Software Framework.
Copyright © 2025 Simplito sp. z o.o.

This file is part of the DeepFellow Software Framework (https://deepfellow.ai).
This software is Licensed under the DeepFellow Free License.

See the License for the specific language governing permissions and
limitations under the License.
*/
const reportWebVitals = (onPerfEntry?: () => void) => {
	if (onPerfEntry && onPerfEntry instanceof Function) {
		import("web-vitals").then(({ onCLS, onINP, onFCP, onLCP, onTTFB }) => {
			onCLS(onPerfEntry);
			onINP(onPerfEntry);
			onFCP(onPerfEntry);
			onLCP(onPerfEntry);
			onTTFB(onPerfEntry);
		});
	}
};

export default reportWebVitals;
