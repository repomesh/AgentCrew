"""
Browser automation service for controlling Chrome browser.

This service provides functionality to navigate web pages, click elements,
scroll content, and extract page information using Chrome DevTools Protocol.
"""

import time
from typing import Dict, Any, Optional, List
from html_to_markdown import convert, ConversionOptions, PreprocessingOptions
import urllib.parse

from html.parser import HTMLParser
import re

from .chrome_manager import ChromeManager
from .element_extractor import (
    extract_clickable_elements,
    extract_input_elements,
    extract_elements_by_text,
    extract_scrollable_elements,
    clean_markdown_images,
    remove_duplicate_lines,
)
from .js_loader import js_loader, JavaScriptExecutor

import PyChromeDevTools
from loguru import logger


class BrowserAutomationService:
    """Service for browser automation using Chrome DevTools Protocol."""

    def __init__(self, debug_port: int = 9222):
        """
        Initialize browser automation service.

        Args:
            debug_port: Port for Chrome DevTools Protocol
        """

        self.debug_port = debug_port
        self.chrome_manager = ChromeManager(debug_port=debug_port)
        self.chrome_interface: Optional[PyChromeDevTools.ChromeInterface] = None
        self._is_initialized = False
        # UUID to XPath mapping for element identification
        self.uuid_to_xpath_mapping: Dict[str, str] = {}
        self._last_page_content: str = ""

    def _find_page_tab(self) -> int:
        """Find the first tab with type 'page' from Chrome's tab list."""
        if self.chrome_interface is None:
            return 0
        self.chrome_interface.get_tabs()
        if self.chrome_interface.tabs:
            for i, tab in enumerate(self.chrome_interface.tabs):
                if tab.get("type") == "page":
                    return i
        return 0

    def _ensure_chrome_running(self, profile: str = "Default"):
        """Ensure Chrome browser is running and connected."""
        if not self._is_initialized:
            self._initialize_chrome(profile)
        if self.chrome_interface:
            tab_index = self._find_page_tab()
            self.chrome_interface.connect(tab=tab_index)

    def _initialize_chrome(self, profile: str = "Default"):
        """Initialize Chrome browser and DevTools connection."""
        try:
            if not self.chrome_manager.is_chrome_running():
                self.chrome_manager.start_chrome_thread(profile)

                if not self.chrome_manager.is_chrome_running():
                    raise RuntimeError("Failed to start Chrome browser")

            time.sleep(2)

            self.chrome_interface = PyChromeDevTools.ChromeInterface(
                host="localhost",
                port=self.debug_port,
                auto_connect=False,
                suppress_origin=True,
            )

            tab_index = self._find_page_tab()
            self.chrome_interface.connect(tab=tab_index)

            self.chrome_interface.Network.enable()
            self.chrome_interface.Page.enable()
            self.chrome_interface.Runtime.enable()
            self.chrome_interface.Emulation.enable()

            self.chrome_interface.DOM.enable()

            self._is_initialized = True

        except Exception as e:
            logger.error(f"Failed to initialize Chrome: {e}")
            self._is_initialized = False
            raise

    def navigate(self, url: str, profile: str = "Default") -> Dict[str, Any]:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to
            profile: Chrome user profile directory name (default: "Default")

        Returns:
            Dict containing navigation result
        """
        try:
            self._ensure_chrome_running(profile)
            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            result = self.chrome_interface.Page.navigate(url=urllib.parse.unquote(url))

            if isinstance(result, tuple) and len(result) >= 2:
                if isinstance(result[0], dict):
                    error_text = result[0].get("result", {}).get("errorText")
                    if error_text:
                        self.chrome_manager.cleanup()
                        self._is_initialized = False
                        self.chrome_interface = None
                        return {
                            "success": False,
                            "error": f"Navigation failed: {error_text}.Please try again",
                            "url": url,
                            "profile": profile,
                        }

            current_url = JavaScriptExecutor.get_current_url(self.chrome_interface)

            return {
                "success": True,
                "message": f"Successfully navigated to {url}",
                "current_url": current_url,
                "url": url,
                "profile": profile,
            }

        except Exception as e:
            logger.error(f"Navigation error: {e}")
            self.chrome_manager.cleanup()
            self._is_initialized = False
            self.chrome_interface = None
            return {
                "success": False,
                "error": f"Navigation failed: {str(e)}. Please try again",
                "url": url,
                "profile": profile,
            }

    def refresh(self) -> Dict[str, Any]:
        """
        Refresh the current page.

        Returns:
            Dict containing refresh result
        """
        try:
            self._ensure_chrome_running()
            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            self.chrome_interface.Page.reload()

            import time

            time.sleep(0.5)

            current_url = JavaScriptExecutor.get_current_url(self.chrome_interface)

            return {
                "success": True,
                "message": "Successfully refreshed the page",
                "current_url": current_url,
            }

        except Exception as e:
            logger.error(f"Refresh error: {e}")
            return {
                "success": False,
                "error": f"Refresh error: {str(e)}",
            }

    def click_element(self, element_uuid: str) -> Dict[str, Any]:
        """
        Click an element using UUID via Chrome DevTools Protocol.

        This method uses CDP's Input.dispatchMouseEvent to simulate real mouse clicks
        by calculating element coordinates and triggering mousePressed/mouseReleased events.

        Args:
            element_uuid: UUID of the element to click (from browser_get_content)

        Returns:
            Dict containing click result
        """
        xpath = self.uuid_to_xpath_mapping.get(element_uuid)
        if not xpath:
            return {
                "success": False,
                "error": f"Element UUID '{element_uuid}' not found. Please use browser_get_content to get current element UUIDs.",
                "uuid": element_uuid,
            }
        try:
            self._ensure_chrome_running()

            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            js_code = js_loader.get_click_element_js(xpath)
            coord_result = JavaScriptExecutor.execute_and_parse_result(
                self.chrome_interface, js_code
            )

            if not coord_result.get("success", False):
                return {
                    "success": False,
                    "error": coord_result.get(
                        "error", "Failed to calculate coordinates"
                    ),
                    "uuid": element_uuid,
                    "xpath": xpath,
                }

            x = coord_result.get("x")
            y = coord_result.get("y")

            if x is None or y is None:
                return {
                    "success": False,
                    "error": "Coordinates not found in calculation result",
                    "uuid": element_uuid,
                    "xpath": xpath,
                }

            time.sleep(0.2)

            self.chrome_interface.Input.dispatchMouseEvent(
                type="mousePressed", x=x, y=y, button="left", clickCount=1
            )

            time.sleep(0.05)

            self.chrome_interface.Input.dispatchMouseEvent(
                type="mouseReleased", x=x, y=y, button="left", clickCount=1
            )

            time.sleep(0.5)

            return {
                "success": True,
                "message": "Element clicked successfully",
                "uuid": element_uuid,
                "xpath": xpath,
                "coordinates": {"x": x, "y": y},
                "elementInfo": coord_result.get("elementInfo", {}),
                "method": "chrome_devtools_protocol",
            }

        except Exception as e:
            logger.error(f"Click error: {e}")
            return {
                "success": False,
                "error": f"Click error: {str(e)}",
                "uuid": element_uuid,
                "xpath": xpath,
            }

    def scroll_to_element(self, element_uuid: str) -> Dict[str, Any]:
        """
        Scroll to bring a specific element into view.

        Args:
            element_uuid: UUID of the element to scroll to

        Returns:
            Dict containing scroll result
        """
        try:
            self._ensure_chrome_running()

            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            xpath = self.uuid_to_xpath_mapping.get(element_uuid)
            if not xpath:
                return {
                    "success": False,
                    "error": f"Element UUID '{element_uuid}' not found. Please use browser_get_content to get current element UUIDs.",
                    "uuid": element_uuid,
                }

            js_code = js_loader.get_scroll_to_element_js(xpath)

            scroll_result = JavaScriptExecutor.execute_and_parse_result(
                self.chrome_interface, js_code
            )

            time.sleep(0.5)

            result_data = {"uuid": element_uuid, "xpath": xpath, **scroll_result}
            return result_data

        except Exception as e:
            logger.error(f"Scroll to element error: {e}")
            return {
                "success": False,
                "error": f"Scroll to element error: {str(e)}",
                "uuid": element_uuid,
            }

    def get_page_content(self) -> Dict[str, Any]:
        """
        Extract page content and clickable elements as markdown.

        Returns:
            Dict containing page content and clickable elements
        """
        try:
            self._ensure_chrome_running()

            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            # Get page document
            _, dom_data = self.chrome_interface.DOM.getDocument(depth=1)

            retry_count = 0

            while (
                not dom_data or len(dom_data) < 1 or not dom_data[0].get("result", None)
            ):
                time.sleep(1)
                _, dom_data = self.chrome_interface.DOM.getDocument(depth=1)
                retry_count += 1
                if retry_count >= 5:
                    break

            result = JavaScriptExecutor.filter_hidden_elements(self.chrome_interface)

            if result.get("success"):
                filtered_html = result.get("html", "")
                logger.info(
                    "Successfully filtered hidden elements using computed styles"
                )

            else:
                # Find HTML node
                html_node = None
                for node in dom_data[0]["result"]["root"]["children"]:
                    if node.get("nodeName") == "HTML":
                        html_node = node
                        break

                if not html_node:
                    return {
                        "success": False,
                        "error": "Could not find HTML node in page",
                    }

                # Get outer HTML
                html_content, _ = self.chrome_interface.DOM.getOuterHTML(
                    nodeId=html_node["nodeId"]
                )
                raw_html = html_content.get("result", {}).get("outerHTML", "")

                if not raw_html:
                    return {"success": False, "error": "Could not extract HTML content"}

                # Filter out hidden elements using JavaScript (doesn't modify page)
                filtered_html = self._filter_hidden_elements(raw_html)

            # Convert HTML to markdown
            # raw_markdown_content = convert_to_markdown(
            #     filtered_html,
            #     source_encoding="utf-8",
            #     strip_newlines=True,
            #     extract_metadata=False,
            #     remove_forms=False,
            #     remove_navigation=False,
            # )
            raw_markdown_content = convert(
                filtered_html,
                ConversionOptions(
                    strip_newlines=True,
                    extract_metadata=False,
                ),
                PreprocessingOptions(
                    remove_navigation=False, remove_forms=False, preset="minimal"
                ),
            )
            if not raw_markdown_content:
                return {"success": False, "error": "Could not convert HTML to markdown"}

            # Clean the markdown content
            cleaned_markdown_content = clean_markdown_images(raw_markdown_content)

            # Remove consecutive duplicate lines
            deduplicated_content = remove_duplicate_lines(cleaned_markdown_content)

            self.uuid_to_xpath_mapping.clear()

            clickable_elements_md = extract_clickable_elements(
                self.chrome_interface, self.uuid_to_xpath_mapping
            )

            input_elements_md = extract_input_elements(
                self.chrome_interface, self.uuid_to_xpath_mapping
            )

            scrollable_elements_md = extract_scrollable_elements(
                self.chrome_interface, self.uuid_to_xpath_mapping
            )

            final_content = (
                deduplicated_content
                + clickable_elements_md
                + input_elements_md
                + scrollable_elements_md
            )

            final_content = final_content.encode("utf-8", "ignore").decode(
                "utf-8", "ignore"
            )

            current_url = JavaScriptExecutor.get_current_url(self.chrome_interface)

            return {
                "success": True,
                "content": final_content,
                "url": current_url,
            }

        except Exception as e:
            logger.error(f"Content extraction error: {e}")
            return {"success": False, "error": f"Content extraction error: {str(e)}"}

    def _filter_hidden_elements(self, html_content: str) -> str:
        """
        Filter out HTML elements that have style='display:none' or aria-hidden='true'.

        Args:
            html_content: Raw HTML content to filter

        Returns:
            Filtered HTML content with hidden elements removed
        """

        class HiddenElementFilter(HTMLParser):
            def __init__(self):
                super().__init__()
                self.filtered_html = []
                self.skip_depth = 0
                self.tag_stack = []

            def handle_starttag(self, tag, attrs):
                # Convert attrs to dict for easier access
                attr_dict = dict(attrs)
                should_hide = False

                if self.skip_depth > 0:
                    if tag in self.tag_stack:
                        self.skip_depth += 1
                    return

                if tag.lower() in ["script", "style", "svg"]:
                    should_hide = True

                # Check for style="display:none" (case insensitive, flexible matching)
                style = attr_dict.get("style", "")
                if style:
                    # Remove spaces and check for display:none
                    style_clean = re.sub(r"\s+", "", style.lower())
                    if (
                        "display:none" in style_clean
                        or "display=none" in style_clean
                        or "visibility:hidden" in style_clean
                    ):
                        should_hide = True

                # Check for aria-hidden="true"
                aria_hidden = attr_dict.get("aria-hidden", "")
                if aria_hidden and aria_hidden.lower() == "true":
                    should_hide = True

                if should_hide:
                    if tag.lower() in ["img", "input", "br", "hr", "meta", "link"]:
                        # Self-closing tags, just skip
                        return
                    self.tag_stack.append(tag)
                    self.skip_depth += 1
                    return

                if self.skip_depth == 0:
                    # Reconstruct the tag with its attributes
                    attr_string = " ".join([f'{k}="{v}"' for k, v in attrs])
                    if attr_string:
                        self.filtered_html.append(f"<{tag} {attr_string}>")
                    else:
                        self.filtered_html.append(f"<{tag}>")

            def handle_endtag(self, tag):
                if self.skip_depth > 0:
                    if tag in self.tag_stack:
                        self.skip_depth -= 1
                        if self.skip_depth == 0:
                            self.tag_stack.remove(tag)
                        return

                if self.skip_depth == 0:
                    self.filtered_html.append(f"</{tag}>")

            def handle_data(self, data):
                if self.skip_depth == 0:
                    self.filtered_html.append(data)

            def handle_comment(self, data):
                if self.skip_depth == 0:
                    self.filtered_html.append(f"<!--{data}-->")

            def handle_entityref(self, name):
                if self.skip_depth == 0:
                    self.filtered_html.append(f"&{name};")

            def handle_charref(self, name):
                if self.skip_depth == 0:
                    self.filtered_html.append(f"&#{name};")

            def get_filtered_html(self):
                return "".join(self.filtered_html)

        try:
            parser = HiddenElementFilter()
            parser.feed(html_content)
            return parser.get_filtered_html()
        except Exception as e:
            logger.warning(f"Error filtering hidden elements: {e}")
            # Return original content if filtering fails
            return html_content

    def cleanup(self):
        """Clean up browser resources."""
        try:
            if self.chrome_manager:
                self.chrome_manager.cleanup()
            self._is_initialized = False
            self.chrome_interface = None
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

    def input_data(self, element_uuid: str, value: str) -> Dict[str, Any]:
        """
        Input data into a form field using UUID by simulating keyboard typing.

        Args:
            element_uuid: UUID of the input element (from browser_get_content)
            value: Value to input into the field

        Returns:
            Dict containing input result
        """
        # Resolve UUID to XPath
        xpath = self.uuid_to_xpath_mapping.get(element_uuid)
        if not xpath:
            return {
                "success": False,
                "error": f"Element UUID '{element_uuid}' not found. Please use browser_get_content to get current element UUIDs.",
                "uuid": element_uuid,
                "input_value": value,
            }
        try:
            self._ensure_chrome_running()

            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            focus_result = JavaScriptExecutor.focus_and_clear_element(
                self.chrome_interface, xpath
            )
            if not focus_result.get("success", False):
                return focus_result

            can_simulate_typing = focus_result.get("canSimulateTyping", False)
            if can_simulate_typing:
                typing_result = JavaScriptExecutor.simulate_typing(
                    self.chrome_interface, value
                )
                if not typing_result.get("success", False):
                    return {
                        **typing_result,
                        "uuid": element_uuid,
                        "xpath": xpath,
                        "input_value": value,
                    }

            JavaScriptExecutor.trigger_input_events(self.chrome_interface, xpath, value)
            time.sleep(1.5)

            return {
                "success": True,
                "message": f"Successfully typed '{value}' using keyboard simulation",
                "uuid": element_uuid,
                "xpath": xpath,
                "input_value": value,
                "typing_method": "keyboard_simulation",
            }

        except Exception as e:
            logger.error(f"Keyboard input simulation error: {e}")
            return {
                "success": False,
                "error": f"Keyboard input simulation error: {str(e)}",
                "uuid": element_uuid,
                "xpath": xpath,
                "input_value": value,
                "typing_method": "keyboard_simulation",
            }

    def dispatch_key_event(self, key: str, modifiers: List[str] = []) -> Dict[str, Any]:
        """
        Dispatch key events using CDP input.dispatchKeyEvent.

        Args:
            key: Key to dispatch (e.g., 'Enter', 'Up', 'Down', 'F1', 'PageUp')
            modifiers: Optional modifiers like 'ctrl', 'alt', 'shift'

        Returns:
            Dict containing dispatch result
        """
        try:
            self._ensure_chrome_running()

            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            return JavaScriptExecutor.dispatch_key_event(
                self.chrome_interface, key, modifiers
            )

        except Exception as e:
            logger.error(f"Key dispatch error: {e}")
            return {
                "success": False,
                "error": f"Key dispatch error: {str(e)}",
                "key": key,
                "modifiers": modifiers,
            }

    def get_elements_by_text(self, text: str) -> Dict[str, Any]:
        """Find elements containing specified text using XPath."""
        try:
            self._ensure_chrome_running()
            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            initial_mapping_count = len(self.uuid_to_xpath_mapping)
            elements_md = extract_elements_by_text(
                self.chrome_interface, self.uuid_to_xpath_mapping, text
            )
            new_mapping_count = len(self.uuid_to_xpath_mapping) - initial_mapping_count

            return {
                "success": True,
                "content": elements_md,
                "text": text,
                "elements_found": new_mapping_count,
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Get elements by text error: {str(e)}",
                "text": text,
            }

    def capture_screenshot(
        self,
        format: str = "png",
        quality: Optional[int] = None,
        clip: Optional[Dict[str, Any]] = None,
        from_surface: bool = True,
        capture_beyond_viewport: bool = False,
    ) -> Dict[str, Any]:
        """
        Capture a screenshot of the current page with colored boxes and UUID labels drawn over identified elements.

        Args:
            format: Image format ("png", "jpeg", or "webp"). Defaults to "png"
            quality: Compression quality from 0-100 (jpeg only). Optional
            clip: Optional region to capture. Dict with x, y, width, height keys
            from_surface: Capture from surface rather than view. Defaults to True
            capture_beyond_viewport: Capture beyond viewport. Defaults to False

        Returns:
            Dict containing the screenshot as base64 image data in the specified format
        """
        try:
            self._ensure_chrome_running()

            if self.chrome_interface is None:
                raise RuntimeError("Chrome interface is not initialized")

            boxes_drawn = False
            if self.uuid_to_xpath_mapping:
                draw_result = JavaScriptExecutor.draw_element_boxes(
                    self.chrome_interface, self.uuid_to_xpath_mapping
                )
                if draw_result.get("success"):
                    boxes_drawn = True
                    logger.info(
                        f"Drew {draw_result.get('count', 0)} element boxes for screenshot"
                    )
                else:
                    logger.warning(
                        f"Failed to draw element boxes: {draw_result.get('error')}"
                    )

            # Prepare parameters for screenshot capture
            screenshot_params = {
                "format": format,
                "fromSurface": from_surface,
                "captureBeyondViewport": capture_beyond_viewport,
            }

            # Add quality parameter only for jpeg format
            if format == "jpeg" and quality is not None:
                screenshot_params["quality"] = quality

            # Add clip parameter if provided
            if clip is not None:
                screenshot_params["clip"] = clip

            # Capture the screenshot
            result = self.chrome_interface.Page.captureScreenshot(**screenshot_params)

            if isinstance(result, tuple) and len(result) >= 2:
                if isinstance(result[1], dict):
                    screenshot_data = result[1].get("result", {}).get("data", "")
                elif isinstance(result[1], list) and len(result[1]) > 0:
                    screenshot_data = result[1][-1].get("result", {}).get("data", "")
                else:
                    return {
                        "success": False,
                        "error": "Invalid response format from screenshot capture",
                    }
            else:
                return {
                    "success": False,
                    "error": "No response from screenshot capture",
                }

            if not screenshot_data:
                return {"success": False, "error": "No screenshot data received"}

            # import base64
            # from datetime import datetime
            #
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # filename = f"screenshot_{timestamp}.{format}"
            #
            # try:
            #     screenshot_bytes = base64.b64decode(screenshot_data)
            #     with open(filename, "wb") as f:
            #         f.write(screenshot_bytes)
            #     logger.info(f"Screenshot saved to {filename} for debugging")
            # except Exception as save_error:
            #     logger.warning(f"Failed to save screenshot to file: {save_error}")
            #
            mime_type_map = {
                "png": "image/png",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
            }
            mime_type = mime_type_map.get(format, "image/png")

            current_url = JavaScriptExecutor.get_current_url(self.chrome_interface)

            if boxes_drawn:
                remove_result = JavaScriptExecutor.remove_element_boxes(
                    self.chrome_interface
                )
                if not remove_result.get("success"):
                    logger.warning(
                        f"Failed to remove element boxes: {remove_result.get('error')}"
                    )

            return {
                "success": True,
                "message": f"Successfully captured screenshot in {format} format",
                "screenshot": {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{screenshot_data}"},
                },
                "format": format,
                "url": current_url,
            }

        except Exception as e:
            logger.error(f"Screenshot capture error: {e}")
            return {"success": False, "error": f"Screenshot capture error: {str(e)}"}

    def __del__(self):
        """Cleanup when service is destroyed."""
        self.cleanup()
