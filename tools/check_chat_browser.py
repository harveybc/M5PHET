"""Browser acceptance against a running manual workbench, not a model benchmark."""
import argparse
import json
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1, reduced_motion="reduce")
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.wait_for_selector(".chat-item")
        page.screenshot(path=str(args.out / "desktop.png"), full_page=True)
        page.get_by_role("button", name="Configurar chat", exact=True).click()
        page.get_by_role("tab", name="Motor", exact=True).click()
        page.screenshot(path=str(args.out / "settings.png"), full_page=True)
        page.get_by_role("button", name="Cerrar configuración").click()
        if args.infer:
            page.get_by_role("button", name="Nuevo chat", exact=True).click()
            page.get_by_role("button", name="Noticia · Clasificación", exact=True).click()
            page.get_by_role("button", name="Enviar pregunta", exact=True).click()
            page.locator('.message.assistant').wait_for(timeout=20000)
            page.locator('.running').wait_for(state='detached', timeout=180000)
            assert page.locator(".message.assistant .tag").first.inner_text() == "OK", page.locator(".message.assistant").inner_text()
            page.screenshot(path=str(args.out / "result.png"), full_page=True)
            before = page.locator(".result-label").inner_text()
            page.reload()
            page.wait_for_selector(".result-label")
            assert page.locator(".result-label").inner_text() == before
        for width, height in ((390, 844), (768, 1024)):
            page.set_viewport_size({"width": width, "height": height})
            page.screenshot(path=str(args.out / f"mobile-{width}.png"), full_page=True)
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"horizontal overflow at {width}"
            prompt = page.locator("#prompt").bounding_box()
            assert prompt and prompt["y"] + prompt["height"] <= height
        page.set_viewport_size({"width": 390, "height": 844})
        page.get_by_role("button", name="Conversaciones", exact=True).click()
        page.screenshot(path=str(args.out / "mobile-sidebar.png"), full_page=True)
        page.get_by_role("button", name="Cerrar conversaciones", exact=True).click(position={"x": 350, "y": 400})
        browser.close()
    assert not errors, errors
    (args.out / "browser-results.json").write_text(json.dumps({"page_errors": errors, "viewports": [1440, 390, 768], "real_inference_requested": args.infer}, indent=2))


if __name__ == "__main__":
    main()
