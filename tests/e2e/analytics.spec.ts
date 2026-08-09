import { expect, test } from "@playwright/test";

// Pins the shape of the two analytics tags this site ships, so a copy edit or a
// careless attribute drop cannot silently change what leaves a visitor's
// browser. Mirrors the approach in a14a-landing's tests/cms-content.spec.ts.
//
// Nothing here talks to the network: the tags are asserted as markup, and the
// tracker script itself is stubbed out in the custom-event test below.

const CHILITRACK_TAG = "script#chilitrack-analytics";
const UMAMI_TAG = "script[src='https://analytics.a14a.org/script.js']";

test.describe("analytics tags", () => {
	test.beforeEach(async ({ page }) => {
		await page.goto("/", { waitUntil: "domcontentloaded" });
	});

	test("ChiliTrack tag carries the mlxgpt.com website id and redaction flags", async ({ page }) => {
		const tag = page.locator(CHILITRACK_TAG);
		await expect(tag).toHaveCount(1);

		await expect(tag).toHaveAttribute("src", "https://ingest.chilitrack.com/script.js");
		await expect(tag).toHaveAttribute("data-website-id", "c432d2ba-2529-4a99-91e5-d07bd4bcbbbe");
		await expect(tag).toHaveAttribute("data-domains", "mlxgpt.com");

		// Umami also runs on this page. side-by-side is what keeps ChiliTrack off
		// window.umami; without it the two products fight over the same handle.
		await expect(tag).toHaveAttribute("data-chili-mode", "side-by-side");

		// Without these the RUM and error-breadcrumb streams ship full URLs,
		// query string and fragment included.
		await expect(tag).toHaveAttribute("data-exclude-search", "true");
		await expect(tag).toHaveAttribute("data-exclude-hash", "true");

		await expect(tag).toHaveAttribute("data-performance", "true");
	});

	test("no stream is enabled that the site does not disclose", async ({ page }) => {
		// mlxgpt.com publishes no privacy policy at all. Session replay and
		// interaction heatmaps record visitor behaviour and must stay off until
		// published copy describes them. If you are turning one of these on,
		// the policy has to land first -- that is the whole point of this test.
		const tag = page.locator(CHILITRACK_TAG);
		await expect(tag).not.toHaveAttribute("data-replay", /.*/);
		await expect(tag).not.toHaveAttribute("data-heatmap", /.*/);
	});

	test("the Umami tag is still present and unchanged", async ({ page }) => {
		const tag = page.locator(UMAMI_TAG);
		await expect(tag).toHaveCount(1);
		await expect(tag).toHaveAttribute("data-website-id", "8f50186e-74cb-4d63-978d-bd4987e04b2b");
	});
});

test("repo CTA clicks are reported to ChiliTrack's own handle", async ({ page }) => {
	// ChiliTrack takes window.chili in side-by-side mode precisely so it does
	// not collide with Umami. This asserts the page sends its custom event to
	// that handle: if it were changed to window.umami the event would be filed
	// with the wrong product, which no amount of tag-attribute checking catches.
	await page.route("https://ingest.chilitrack.com/script.js", (route) =>
		route.fulfill({ status: 200, contentType: "text/javascript", body: "" }),
	);
	await page.route("https://analytics.a14a.org/script.js", (route) =>
		route.fulfill({ status: 200, contentType: "text/javascript", body: "" }),
	);

	await page.addInitScript(() => {
		const calls: Array<{ name: string; data?: Record<string, unknown> }> = [];
		(window as unknown as Record<string, unknown>).__chiliCalls = calls;
		(window as unknown as Record<string, unknown>).chili = {
			track: (name: string, data?: Record<string, unknown>) => {
				calls.push({ name, data });
			},
		};
	});

	await page.goto("/", { waitUntil: "domcontentloaded" });

	// Do not actually leave the page for GitHub during the test.
	await page.locator('nav a[href*="github.com/a14a-org/mlxgpt"]').evaluate((element) => {
		element.addEventListener("click", (event) => event.preventDefault());
	});
	await page.locator('nav a[href*="github.com/a14a-org/mlxgpt"]').click();

	const calls = await page.evaluate(
		() =>
			(window as unknown as Record<string, unknown>).__chiliCalls as Array<{
				name: string;
				data?: Record<string, unknown>;
			}>,
	);

	expect(calls.map((call) => call.name)).toContain("repo_click");
	expect(calls[0]?.data?.location).toBe("nav");
});
