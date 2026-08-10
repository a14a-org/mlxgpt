import { expect, test } from "@playwright/test";

// ChiliTrack only redacts URLs when it is told to. Without these two
// attributes the tag sends location.href verbatim — query string and fragment
// included — from the RUM payload and from error breadcrumbs. Pin them so a
// future edit to the tag cannot silently widen what leaves the browser.
test("the ChiliTrack tag strips the query string and the fragment", async ({ page }) => {
	await page.goto("/", { waitUntil: "domcontentloaded" });

	const tag = page.locator("script#chilitrack-analytics");
	await expect(tag).toHaveCount(1);
	await expect(tag).toHaveAttribute("data-exclude-search", "true");
	await expect(tag).toHaveAttribute("data-exclude-hash", "true");
});
