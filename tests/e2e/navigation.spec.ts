import { expect, test } from "@playwright/test";

// Real user-flow tests for the single-page mlxgpt.com landing site. These
// assert structural behavior (anchor navigation, section presence, the GitHub
// CTA) rather than brittle copy, so they stay green across content edits.

test.beforeEach(async ({ page }) => {
	await page.goto("/", { waitUntil: "domcontentloaded" });
});

test("nav anchor links point at sections that exist on the page", async ({ page }) => {
	const nav = page.locator("nav");
	await expect(nav).toBeVisible();

	const anchorLinks = nav.locator('a[href^="#"]');
	const count = await anchorLinks.count();
	expect(count, "expected in-page nav anchors").toBeGreaterThan(0);

	for (let i = 0; i < count; i++) {
		const href = await anchorLinks.nth(i).getAttribute("href");
		if (!href || href === "#") continue;
		const id = href.slice(1);
		// Every nav target must resolve to a real element on the page.
		await expect(page.locator(`#${id}`)).toHaveCount(1);
	}
});

test("clicking a nav anchor scrolls the matching section into view", async ({ page }) => {
	// The site intercepts anchor clicks (preventDefault) and smooth-scrolls via
	// scrollIntoView, so the URL hash does not change — assert on viewport
	// position, which is what the user actually experiences.
	const section = page.locator("#architecture");
	await expect(section).not.toBeInViewport();
	await page.locator('nav a[href="#architecture"]').click();
	await expect(section).toBeInViewport({ timeout: 5000 });
});

test("GitHub CTA opens the project repository in a new tab", async ({ page }) => {
	const githubLink = page.locator('nav a[href*="github.com/a14a-org/mlxgpt"]');
	await expect(githubLink).toBeVisible();
	// External links are expected to open in a new tab safely.
	await expect(githubLink).toHaveAttribute("target", "_blank");
	await expect(githubLink).toHaveAttribute("rel", /noopener/);
});

test("the four key content sections are present", async ({ page }) => {
	for (const id of ["metrics", "architecture", "log", "next"]) {
		await expect(page.locator(`#${id}`)).toHaveCount(1);
	}
});
