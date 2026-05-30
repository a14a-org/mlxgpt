import { expect, test } from "@playwright/test";

// Content-agnostic smoke: every served asset must respond without an HTTP
// error and (for the HTML page) without throwing an uncaught client exception.
// Kept text-agnostic so copy edits do not break the suite — it verifies the
// site still boots and routes still resolve.
const routes = ["/", "/robots.txt", "/sitemap.xml"];

for (const path of routes) {
	test(`smoke: ${path} responds without errors`, async ({ page }) => {
		const pageErrors: string[] = [];
		page.on("pageerror", (err) => pageErrors.push(String(err)));

		const response = await page.goto(path, { waitUntil: "domcontentloaded" });

		expect(response, `no response for ${path}`).not.toBeNull();
		expect(response?.status(), `HTTP status for ${path}`).toBeLessThan(400);
		expect(pageErrors, `uncaught exceptions on ${path}`).toEqual([]);
	});
}

test("home page renders a visible body and a non-empty title", async ({ page }) => {
	await page.goto("/", { waitUntil: "domcontentloaded" });
	await expect(page.locator("body")).toBeVisible();
	const title = await page.title();
	expect(title.length).toBeGreaterThan(0);
});
