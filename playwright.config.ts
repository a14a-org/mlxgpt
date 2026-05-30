import { defineConfig, devices } from "@playwright/test";

// E2E config for the static mlxgpt.com site. The site is plain HTML served by
// nginx in production (see Dockerfile); for tests we serve site/ statically on
// port 4173 and run the smoke suite against it.
const PORT = 4173;

export default defineConfig({
	testDir: "./tests/e2e",
	fullyParallel: true,
	forbidOnly: !!process.env.CI,
	retries: process.env.CI ? 2 : 0,
	workers: process.env.CI ? 1 : undefined,
	reporter: process.env.CI ? [["html", { open: "never" }], ["list"]] : "list",

	use: {
		baseURL: `http://localhost:${PORT}`,
		trace: "on-first-retry",
	},

	projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],

	webServer: {
		command: `bunx serve site --listen ${PORT} --no-clipboard --single`,
		url: `http://localhost:${PORT}`,
		reuseExistingServer: !process.env.CI,
		timeout: 120000,
	},
});
