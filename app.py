"""
Gradio Frontend for DataArc Synthetic Data Generation System
"""
import gradio as gr
import yaml
import os
import tempfile
import shutil
from pathlib import Path
import logging
from datetime import datetime
from typing import Optional, Tuple, List
import base64

from sdgsystem.configs.config import SDGSConfig
from sdgsystem.pipeline import Pipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogCapture(logging.Handler):
    """Custom logging handler to capture logs for Gradio display"""
    def __init__(self):
        super().__init__()
        self.logs = []
        self.last_index = 0

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

    def get_logs(self):
        return "\n".join(self.logs)

    def get_new_logs(self):
        """Get only new logs since last call"""
        new_logs = self.logs[self.last_index:]
        self.last_index = len(self.logs)
        return "\n".join(new_logs) if new_logs else ""

    def clear(self):
        self.logs = []
        self.last_index = 0


# Global log capture handler
log_capture = LogCapture()
log_capture.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(log_capture)

# Add to root logger to capture ALL logs from any module
root_logger = logging.getLogger()
root_logger.addHandler(log_capture)
root_logger.setLevel(logging.INFO)


# Load and encode logo image
def get_logo_base64():
    """Load logo image and convert to base64"""
    logo_path = os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')
    try:
        with open(logo_path, 'rb') as f:
            logo_data = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{logo_data}"
    except Exception as e:
        logger.warning(f"Failed to load logo: {e}")
        return ""

LOGO_BASE64 = get_logo_base64()


CUSTOM_CSS = """
:root {
    /* Carbon Blue Palette */
    --carbon-blue-900: #021c50;
    --carbon-blue-800: #032d80;
    --carbon-blue-700: #054ada;
    --carbon-blue-600: #524AC9;
    --carbon-blue-500: #4589ff;
    --carbon-blue-400: #78a9ff;
    --carbon-blue-300: #a6c8ff;
    --carbon-green-500: #1f645b;
    --carbon-gray-100: #f4f4f4;
    --carbon-gray-200: #e0e0e0;
    --carbon-gray-300: #c6c6c6;
    --carbon-gray-400: #a8a8a8;
    --carbon-gray-500: #8d8d8d;
    --carbon-gray-600: #6f6f6f;
    --carbon-gray-900: #161616;
    --border-subtle: rgba(0, 0, 0, 0.08);
    --card-shadow: 0 20px 60px rgba(8, 19, 51, 0.14);

    /* Override Gradio Soft theme to use Carbon Blue */
    --primary-50: #edf5ff !important;
    --primary-100: #d0e2ff !important;
    --primary-200: #a6c8ff !important;
    --primary-300: #78a9ff !important;
    --primary-400: #4589ff !important;
    --primary-500: #524AC9 !important;
    --primary-600: #0353e9 !important;
    --primary-700: #0043ce !important;
    --primary-800: #002d9c !important;
    --primary-900: #001d6c !important;
    --primary-950: #001141 !important;

    --secondary-50: #edf5ff !important;
    --secondary-100: #d0e2ff !important;
    --secondary-200: #a6c8ff !important;
    --secondary-300: #78a9ff !important;
    --secondary-400: #4589ff !important;
    --secondary-500: #524AC9 !important;
    --secondary-600: #0353e9 !important;
    --secondary-700: #0043ce !important;
    --secondary-800: #002d9c !important;
    --secondary-900: #001d6c !important;
    --secondary-950: #001141 !important;

    /* Override color accent */
    --color-accent: #524AC9 !important;
    --color-accent-soft: #edf5ff !important;
    --border-color-accent: #78a9ff !important;
    --link-text-color: #524AC9 !important;
    --link-text-color-active: #0353e9 !important;
    --link-text-color-hover: #0043ce !important;
    --link-text-color-visited: #4589ff !important;

    /* Override button colors */
    --button-primary-background-fill: #524AC9 !important;
    --button-primary-background-fill-hover: #0353e9 !important;
    --button-primary-border-color: #524AC9 !important;
    --button-primary-border-color-hover: #0353e9 !important;

    /* Override checkbox colors */
    --checkbox-background-color-selected: #524AC9 !important;
    --checkbox-border-color-focus: #524AC9 !important;
    --checkbox-border-color-selected: #524AC9 !important;
    --checkbox-label-background-fill-selected: #524AC9 !important;

    /* Override slider color */
    --slider-color: #524AC9 !important;

    /* Override input focus color */
    --input-border-color-focus: #524AC9 !important;

    /* Override loader color */
    --loader-color: #524AC9 !important;

    /* Override block label colors */
    --block-label-background-fill: transparent !important;
    --block-label-text-color: #161616 !important;
    --block-title-background-fill: transparent !important;
    --block-title-text-color: #161616 !important;

    --form-gap-width: 0px;
    --layout-gap: 0px;
    --block-border-width: 0px;
}

/* Universal icon style for iconfont symbol */
.icon {
    width: 1em;
    height: 1em;
    vertical-align: -0.15em;
    fill: currentColor;
    overflow: hidden;
}

.gr-group > .styler { background: transparent !important; }
.gr-form { border: none !important; box-shadow: none !important; }
* {
    font-family: 'IBM Plex Sans', 'Source Sans 3', system-ui, -apple-system, BlinkMacSystemFont, 'PingFang SC', 'Microsoft YaHei', sans-serif;
}

body, .gradio-container {
    background: #FFFFFF !important;
    color: var(--carbon-gray-900);
}

gradio-app {
    font-family: inherit;
    background: transparent;
}

footer,
.footer,
gradio-app footer,
.gradio-container footer,
[class*="footer"] {
    display: none !important;
}

.carbon-body {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

#drawer_portal,
#drawer_portal .wrap,
#drawer_portal .html-container {
    height: 0;
    overflow: visible;
}

#drawer_overlay {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.55);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
    z-index: 1100;
}

#drawer_overlay.active {
    opacity: 1;
    pointer-events: all;
}

.carbon-header {
    width: 100%;
    color: white;
    background: white;
}

.carbon-header__top {
    width: 100%;
    background: #1E1B4A;
    box-shadow: inset 0 -1px rgba(255, 255, 255, 0.08);
}

.carbon-header__layout {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 32px;
    padding: 10px 20px !important;
    max-width: 1440px !important;
    margin: 0 auto !important;
}

.carbon-header__left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.brand-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

.brand-logo {
    height: 20px;
    width: auto;
    display: block;
    object-fit: contain;
    max-width: 100%;
}

.brand-text {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 12px;
    font-weight: 600;
    line-height: 1em;
    letter-spacing: 0.013em;
    color: #FFFFFF;
}

.carbon-header__nav {
    flex: 1;
    display: flex;
    justify-content: center;
}

.header-nav {
    display: inline-flex;
    align-items: center;
    gap: 0;
    padding: 0;
    border-radius: 0;
    background: transparent;
    box-shadow: none;
}

.header-nav__item {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 10px;
    border-radius: 0;
    font-weight: 600;
    font-size: 16px;
    line-height: 0.75em;
    letter-spacing: 0.01em;
    color: #F8F8F8;
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease, opacity 0.2s ease;
    border: none;
    background: transparent;
}

.header-nav__item .icon {
    font-size: 18px;
    vertical-align: middle;
}

/* 顶部导航非激活状态 - 文字颜色变淡 */
.carbon-header__top .header-nav__item {
    color: rgba(248, 248, 248, 0.5);
}

/* 顶部导航激活状态 - 恢复完全白色 */
.carbon-header__top .header-nav__item.active {
    background: transparent;
    color: #F8F8F8;
    box-shadow: none;
}

.header-nav__item.active {
    background: transparent;
    color: #F8F8F8;
    box-shadow: none;
}

.header-nav__item.disabled {
    color: rgba(255, 255, 255, 0.6);
    opacity: 1;
    cursor: not-allowed;
}

.header-nav__item:focus-visible,
.carbon-btn:focus-visible {
    outline: 2px solid rgba(255, 255, 255, 0.7);
    outline-offset: 2px;
}

.carbon-header__right {
    display: flex;
    align-items: center;
    gap: 12px;
}

.carbon-header__bottom {
    width: 100%;
    background: white;
    box-shadow: 0 0 8px rgba(0, 0, 0, 0.08);
    padding: 0;
    transition: opacity 0.3s ease, transform 0.3s ease;
    border-bottom: 1px solid #EEEEEE;
}

.carbon-header__bottom .carbon-header__layout {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 10px 0 !important;
}

.carbon-header__bottom .header-nav {
    background: transparent;
    box-shadow: none;
    padding: 0;
    gap: 2px;
    display: inline-flex;
    width: auto;
}

.carbon-header__bottom .header-nav__item {
    color: #7B7A8D;
    padding: 24px 36px;
    border-radius: 10px;
    background: white;
    position: relative;
    flex-direction: row;
    align-items: center;
    gap: 7px;
    font-size: 16px;
    font-weight: 400;
    line-height: 0.9876em;
    letter-spacing: 0.01em;
    height: 40px;
    flex: 0 0 auto;
    justify-content: flex-start;
    white-space: nowrap;
    transition: background-color 0.15s ease, box-shadow 0.15s ease, color 0.15s ease;
}

.carbon-header__bottom .header-nav__item.clickable {
    cursor: pointer;
}

.carbon-header__bottom .header-nav__item.disabled,
.carbon-header__bottom .header-nav__item:not(.clickable) {
    cursor: not-allowed;
    opacity: 0.6;
}

.carbon-header__bottom .header-nav__item.clickable:hover {
    background: #f8f8f8;
}

.carbon-header__bottom .header-nav__item .nav-dot {
    width: 24px;
    height: 24px;
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.carbon-header__bottom .header-nav__item .nav-dot .iconfont {
    position: absolute;
    font-size: 24px;
    color: #7B7A8D;
    transition: color 0.15s ease;
}

.carbon-header__bottom .header-nav__item .nav-dot .nav-dot-number {
    position: relative;
    z-index: 1;
    font-family: 'Source Sans 3', sans-serif;
    font-size: 16px;
    font-weight: 400;
    line-height: 1;
    color: #7B7A8D;
    transition: color 0.15s ease;
}

.carbon-header__bottom .header-nav__item.active {
    background: rgba(82, 74, 201, 0.05);
    color: #000000;
    box-shadow: none;
}

.carbon-header__bottom .header-nav__item.active .nav-dot .iconfont {
    color: #000000;
}

.carbon-header__bottom .header-nav__item.active .nav-dot .nav-dot-number {
    color: #000000;
}

.carbon-btn {
    border: none;
    border-radius: 0;
    padding: 12px 20px;
    font-size: 15px;
    font-weight: 500;
    line-height: 1;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    transition: background 0.2s ease, opacity 0.2s ease;
}

.carbon-btn.config,
#header_config_btn {
    background: transparent !important;
    color: white !important;
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    padding: 0px 18px;
    height: 48px;
}

.carbon-btn.config .iconfont,
#header_config_btn .iconfont,
#header_config_btn span.iconfont {
    color: white !important;
}

.carbon-btn.generate {
    background: var(--carbon-blue-600);
    color: white;
}

.carbon-btn.generate.disabled,
.carbon-btn.generate:disabled {
    background: rgba(255, 255, 255, 0.12);
    color: rgba(255, 255, 255, 0.45);
    cursor: not-allowed;
    border: 1px solid transparent;
}

.carbon-btn.drawer-trigger {
    background: rgba(255, 255, 255, 0.12);
    color: white;
    border-radius: 8px;
    padding: 10px 16px;
    font-size: 14px;
}

.carbon-btn.drawer-trigger .iconfont {
    font-size: 16px;
}

.carbon-shell {
    width: 100%;
    display: flex;
    justify-content: center;
    padding: 0;
    margin-top: 16px;
}

.carbon-content {
    width: 100%;
    max-width: 50%;
    display: flex;
    flex-direction: column;
    gap: 18px;
    margin: 0 auto 29px auto;
    padding: 0 24px;
}

/* Override Gradio default max width to allow full-width layout */
.gradio-container,
.gradio-container > div,
.gradio-container .main {
    max-width: none !important;
    width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Force full-width layout - override Gradio default restrictions */
gradio-app,
gradio-app > div,
.main,
main,
main.fillable,
main.app,
main.svelte-18evea3,
.app.svelte-18evea3 {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Remove default container constraints */
.contain {
    max-width: none !important;
    padding: 0 !important;
}

/* Ensure fillable containers span full width */
.fillable.svelte-18evea3.svelte-18evea3:not(.fill_width),
.fillable.svelte-18evea3,
.fillable.svelte-18evea3.app,
.app.fillable,
.app.svelte-18evea3,
.app.svelte-18evea3:not(.fill_width) {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

@media (min-width: 1024px) {
    .fillable.svelte-18evea3.svelte-18evea3:not(.fill_width) {
        max-width: none !important;
        width: 100% !important;
    }
}

/* Strip default padding/margins from Gradio wrapper nodes */
.gradio-container .padding,
.gradio-container .panel,
.gradio-container .prose,
.gradio-container .prose .md,
.gradio-container .html-container,
.gradio-container .html-container.padding,
.gradio-container .block:not(.task-type-container),
.gradio-container .block > .wrap:not(.task-type-options),
.gradio-container .gradio-row,
.gradio-container .gradio-column,
.gradio-container .gr-block,
.gradio-container .form,
.gradio-container .gr-form,
.gradio-container .form > .block:not(.task-type-container),
.gradio-container .gr-form > .block:not(.task-type-container) {
    padding: 0 !important;
    margin: 0 !important;
}


#main_layout {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
    margin-top: 20px;
}

#config_section {
    background: white;
    border-radius: 4px;
    width: 50%;
    max-width: 840px;
    min-width: 600px;
    padding-bottom: 90px;
}

.form-section {
    border: none !important;
    border-radius: 0;
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
}

/* Remove all borders between form fields */
.gradio-container .block:not(.task-type-container),
.gradio-container .gr-block,
.gradio-container .gr-box,
.gradio-container .gr-input,
.gradio-container .gr-form > *,
.gradio-container .form > *,
.step-content > .block:not(.task-type-container),
.step-content > .gr-block {
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
    border-left: none !important;
    border-right: none !important;
    box-shadow: none !important;
}

/* Remove container borders */
.gradio-container .container,
.gradio-container .gr-panel,
.gradio-container .panel,
.gradio-container .gr-group:not(.task-type-container):not(.file-upload-container),
.gradio-container [class*="group"]:not(.task-type-container):not(.file-upload-container) {
    border: none !important;
    box-shadow: none !important;
}

/* Remove borders from input wrappers */
.gradio-container .wrap,
.gradio-container .input-wrap,
.gradio-container > div {
    border: none !important;
}

/* Fully hide placeholder center/full wraps that Gradio injects */
.gradio-container .wrap.center.full.hide,
.gradio-container .wrap.center.full.svelte-btia7y.hide {
    display: none !important;
    height: 0 !important;
    width: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

/* Remove borders from parent containers of inputs */
.gradio-container .wrap.svelte-1116j96,
.gradio-container .wrap.default,
.gradio-container [class*="wrap"] {
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
    background-color: white !important;
}

/* Ensure no borders on label containers */
.gradio-container label,
.gradio-container .label-wrap {
    border: none !important;
    border-bottom: none !important;
}

/* Remove Gradio-specific separators and borders */
.gradio-container [class*="svelte"],
.gradio-container .gr-padded,
.gradio-container .gap {
    border: none !important;
}

/* Target specific Gradio component wrappers */
.gradio-container > div > div,
.gradio-container .component-wrapper {
    border: none !important;
}

/* Aggressive border removal for all potential Gradio wrappers */
.gradio-container *:not(.task-type-card):not(.task-type-options):not(input):not(textarea):not(select):not(button):not(.file-upload-container):not(.file-upload-container *):not(.progress-card__badge):not(.output_files_section):not(.carbon-header__layout):not(.carbon-header__bottom):not(.iconfont):not(.output_files_section) {
    border-top: none !important;
    border-bottom: none !important;
}

/* Step content spacing - 35px gap between form items from Figma */
.step-content > * {
    margin-bottom: 35px !important;
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
}

.step-content > *:last-child {
    margin-bottom: 0 !important;
}

/* Specific spacing for form field blocks */
.step-content > .block,
.step-content > .html-container,
.step-content > .gr-group {
    margin-bottom: 35px !important;
}

.step-content > .block:last-child,
.step-content > .html-container:last-child,
.step-content > .gr-group:last-child {
    margin-bottom: 0 !important;
}

.form-field {
    display: flex;
    flex-direction: column;
    gap: 16px;
    margin-bottom: 16px !important;
}

.form-field:last-child {
    margin-bottom: 0 !important;
}

.form-field .field-label-info {
    margin-bottom: 0 !important;
}

/* Ensure clear separation between labels, inputs, and neighboring fields */
.step-content .form-field label,
.step-content .block > label,
.step-content .gr-group > label,
.step-content label.block-label {
    display: inline-block;
    margin-bottom: 10px !important;
}

.step-content .form-field + .form-field,
.step-content .form-field + .block,
.step-content .block + .form-field,
.step-content .block + .block,
.step-content .gr-group + .gr-group,
.step-content .gr-group + .form-field {
    margin-top: 16px !important;
}

/* Remove all borders between components in all steps */
.step-content > .block,
.step-content > .html-container,
.step-content .block + .block,
.step-content .html-container + .block,
.step-content .block + .html-container,
.step-content > div {
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
}

.step-content {
    display: none;
    padding: 24.89px 24.89px 0px 24.89px;
}

.step-content.active {
    display: block;
    animation: fadeIn 0.24s ease;
}

.step-content h3 {
    font-size: 16px;
    letter-spacing: 0.4px;
    color: #161616;
    margin-bottom: 20px;
}

.wizard-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
}

.wizard-stack {
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.wizard-row {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
}

.wizard-row > * {
    flex: 1 1 260px;
}

.step-navigation {
    display: flex;
    justify-content: space-around;
    align-items: center;
    gap: 18px;
    height: 70.889px;
    background: white;
    box-shadow: 0 -1px 4px 0 rgba(0, 0, 0, 0.08);
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    width: 100%;
    z-index: 1000;
    padding: 0 24px;
}

#btn_previous,
#btn_next,
#btn_submit {
    flex: 1;
    max-width: 220px;
    border-radius: 8px;
}

#btn_previous button,
#btn_next button,
#btn_submit button {
    width: 100%;
    border-radius: 8px;
    height: 100%;
    min-height: 48px;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 16px;
    font-weight: 500;
    line-height: 1.2em;
    letter-spacing: 0;
    border: 0.889px solid #E0E0E0;
    transition: all 0.11s cubic-bezier(0.2, 0, 0.38, 0.9);
    padding: 12px 22px;
}

.step-navigation button {
    border-radius: 8px !important;
}

#btn_previous button {
    background: white;
    color: #A4A4A4;
    opacity: 1;
}

#btn_previous button:disabled {
    opacity: 1;
    color: #A4A4A4;
    cursor: not-allowed;
}

#btn_next button,
#btn_submit button {
    border: none;
    background: #524AC9;
    color: white;
}

#btn_next button:hover,
#btn_submit button:hover {
    background: #6B5FD9;
}

#btn_submit button:disabled {
    background: #d1d1d1;
    color: #8d8d8d;
}

#btn_previous {
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
}

#btn_submit {
    display: none;
}

/* Input field wrapper spacing */
.gradio-container .block {
    gap: 10.5px;
}

/* Custom label with icon */
.field-label-with-icon {
    display: flex;
    align-items: center;
    gap: 4px;
    background: #FFFFFF !important;
}

.field-label-text {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 14px;
    line-height: 1.2;
    color: #161616;
}

.field-label-required {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 15px;
    line-height: 1.2;
    color: #DA1E28;
}

.field-label-info {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 11px;
    line-height: 1.27;
    letter-spacing: 0.32px;
    color: #525252;
    margin-bottom: 0;
    background: #FFFFFF !important;
}

.gradio-container label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 14px;
    color: #161616 !important;
    letter-spacing: 0;
    line-height: 1.2;
    margin-bottom: 0;
    background: transparent !important;
    border: none !important;
    border-bottom: none !important;
}

/* Remove auto-added required marker */
.gradio-container label::after {
    content: none !important;
}

/* Add required marker for specific fields using CSS */
.gradio-container label span[data-testid="block-info"]::after {
    content: ' *';
    color: #DA1E28;
    font-weight: 500;
    font-size: 15px;
}

/* Remove marker for optional fields */
.optional-field label span[data-testid="block-info"]::after,
.gradio-container label span.optional-label::after {
    content: none !important;
}

/* Add required marker for custom HTML labels */
.field-label-with-icon.required .field-label-text::after {
    content: ' *';
    color: #DA1E28;
    font-weight: 500;
    font-size: 15px;
}

/* Style for label span container */
.gradio-container label span {
    background: transparent !important;
    color: #161616 !important;
    border: none !important;
    border-bottom: none !important;
}

/* Ensure all Gradio label classes have black text and no background */
.gradio-container .block-label,
.gradio-container .block-title,
.gradio-container label.block-label,
.gradio-container label.block-title,
.gradio-container .label-wrap,
.gradio-container .label-wrap span {
    background: transparent !important;
    color: #161616 !important;
    border: none !important;
    border-bottom: none !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .wrap:has(> input),
.gradio-container .wrap:has(> textarea) {
    border-radius: 4px !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    background: #FAFAFF !important;
    padding: 12px 16px !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 16px;
    line-height: 1.3em;
    color: #161616;
    box-shadow: none !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}

.gradio-container select {
    border-radius: 6px !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
    background: #FAFAFF !important;
    padding: 14px 16px !important;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 16px;
    line-height: 1.3em;
    color: #000000;
    box-shadow: none !important;
    transition: border-color 0.12s ease, background 0.12s ease;
}

.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #999999;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
    background: #FFFFFF !important;
    border: 2px solid #524AC9 !important;
    outline: none !important;
    color: #161616;
}

/* Preserve white cards for form group backgrounds */
.step-content .form-field,
.step-content .block,
.step-content .gr-group,
.step-content .html-container,
.step-content .form,
.step-content .styler,
.step-content .container,
.step-content .wrap,
.step-content .padded,
.step-content .auto-margin {
    background: #FFFFFF !important;
}

/* Ensure label HTML wrappers carry spacing even when nested inside prose blocks */
.step-content .html-container.padding,
.step-content .html-container .prose {
    margin-bottom: 16px !important;
    background: #FFFFFF !important;
    display: block;
}

.step-content .field-label-with-icon {
    margin-bottom: 10px !important;
}

/* Remove padding from dropdown/select components */
.gradio-container .wrap-inner,
.gradio-container .secondary-wrap,
.gradio-container .icon-wrap,
.gradio-container [class*="wrap-inner"],
.gradio-container [class*="secondary-wrap"] {
    padding: 0 !important;
}

/* Style dropdown arrow container */
.gradio-container .icon-wrap {
    padding: 0 !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 18px !important;
    height: 18px !important;
    background: transparent !important;
}

/* Hide default dropdown arrow */
.gradio-container input[role="listbox"] + .icon-wrap svg {
    display: none !important;
}

/* Add custom dropdown arrow using iconfont class */
.gradio-container input[role="listbox"] + .icon-wrap::before {
    font-family: "iconfont" !important;
    font-size: 18px;
    color: #1E1B4A;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    transform: rotate(90deg);
    transition: transform 0.2s ease;
    pointer-events: none;
}

/* Apply the icon class content */
.gradio-container input[role="listbox"] + .icon-wrap {
    font-family: "iconfont" !important;
}

.gradio-container input[role="listbox"] + .icon-wrap::before {
    content: "\e64f";
}

/* Rotate arrow when dropdown is open */
.gradio-container .open input[role="listbox"] + .icon-wrap::before,
.gradio-container input[role="listbox"][aria-expanded="true"] + .icon-wrap::before,
.gradio-container .expanded input[role="listbox"] + .icon-wrap::before,
.gradio-container .wrap.open .icon-wrap::before,
.gradio-container .wrap[aria-expanded="true"] .icon-wrap::before,
.gradio-container .wrap.dropdown-open .icon-wrap::before {
    transform: rotate(-90deg);
}

/* Dropdown wrapper styling to match Figma */

.gradio-container select,
.gradio-container input[role="listbox"],
.gradio-container div[role="listbox"],
.gradio-container .wrap:has(> select),
.gradio-container .wrap:has(> input[role="listbox"]),
.gradio-container .wrap:has(> div[role="listbox"]) {
    border-radius: 6px !important;
    background: #FAFAFF !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
}

/* Dropdown wrapper - only for dropdown containers */
.gradio-container .wrap:has(> input[role="listbox"]) {
    border-radius: 6px !important;
    background: #FAFAFF !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important;
}

/* Dropdown input styling - keep only internal padding for the input itself */
.gradio-container input[role="listbox"],
.gradio-container div[role="listbox"] {
    padding: 14px 16px !important;
    margin: 0 !important;
    background: #FAFAFF !important;
    border: none !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 400 !important;
    line-height: 1.3em !important;
    color: #000000 !important;
}

/* Dropdown options container */
.gradio-container .dropdown-content,
.gradio-container [class*="options"],
.gradio-container .options-wrap {
    background: #FFFFFF !important;
    border-radius: 4px !important;
    box-shadow: 0px 0px 4px 0px rgba(0, 0, 0, 0.08) !important;
    border: none !important;
    margin-top: 4px !important;
}

/* Dropdown option items */
.gradio-container .dropdown-content li,
.gradio-container [class*="options"] li,
.gradio-container .options-wrap li,
.gradio-container .dropdown-item {
    padding: 12px 16px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 16px !important;
    font-weight: 400 !important;
    line-height: 1.3em !important;
    color: #000000 !important;
    background: #FFFFFF !important;
    transition: background 0.15s ease !important;
}

.gradio-container .dropdown-content li:hover,
.gradio-container [class*="options"] li:hover,
.gradio-container .options-wrap li:hover,
.gradio-container .dropdown-item:hover {
    background: #F8F8F8 !important;
}

/* Help text styling to match Figma */
.gradio-container .gr-form label span:last-child,
.gradio-container .block label span:last-child,
.gradio-container .info {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 11px;
    line-height: 1.27em;
    letter-spacing: 0.32px;
    color: #525252 !important;
    font-weight: 400;
    background: transparent !important;
}

/* Task Type Radio - Card Style */
.task-type-container,
.task-type-container .wrap,
.task-type-container > div,
#task_type_cards {
    background: #FFFFFF !important;
    border: none !important;
    padding: 0 !important;
    padding-top: 10px !important;
    box-shadow: none !important;
    pointer-events: auto !important;
}

/* Remove borders around HTML blocks (like Task Type title) */
.gradio-container .html-container,
.step-content > .html-container {
    border: none !important;
    border-top: none !important;
    border-bottom: none !important;
}

/* Remove borders between adjacent components in step content */
#step_2 > *,
#step_2 > div,
#step_2 > .block,
#step_2 .html-container + *,
#task_type_cards + * {
    border-top: none !important;
}

.task-type-options {
    display: flex;
    gap: 13px;
    width: 100%;
    background: transparent;
    position: relative;
    z-index: 1;
}

.task-type-card {
    flex: 1;
    background: #FFFFFF;
    border: 1.78px solid #E0E0E0;
    border-radius: 10px;
    padding: 20px !important;
    cursor: pointer;
    transition: all 0.2s ease;
    position: relative;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    gap: 10.5px;
    pointer-events: auto !important;
    user-select: none;
    z-index: 2;
}

.task-type-card:hover {
    border-color: #524AC9;
}

.task-type-card.disabled {
    opacity: 0.5;
    cursor: not-allowed !important;
    pointer-events: none !important;
}

.task-type-card.disabled:hover {
    border-color: #E0E0E0;
}

.task-type-card.active {
    background: rgba(82, 74, 201, 0.05);
    border: 1.78px solid #524AC9;
}

.coming-soon-badge {
    display: inline-block;
    background: #FFA500;
    color: #FFFFFF;
    font-size: 8px;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 3px;
    margin-left: 6px;
    vertical-align: middle;
    letter-spacing: 0.05em;
}

.task-type-icon {
    width: 35px;
    height: 35px;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 21px;
    flex-shrink: 0;
    pointer-events: none;
}

.task-type-card.active .task-type-icon {
    background: rgba(82, 74, 201, 0.15);
    border-color: transparent;
}

.task-type-content {
    display: flex;
    flex-direction: column;
    gap: 3.5px;
    flex: 1;
    pointer-events: none;
}

.task-type-title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 600;
    font-size: 12.25px;
    line-height: 1.29;
    letter-spacing: 0.0131em;
    color: #161616;
    pointer-events: none;
}

.task-type-card.active .task-type-title {
    color: #524AC9;
}

.task-type-desc {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 10.5px;
    line-height: 1.4;
    letter-spacing: 0.0152em;
    color: #525252;
    pointer-events: none;
}

.task-type-card.active .task-type-desc {
    color: #524AC9;
}

.task-type-check {
    position: absolute;
    top: 12.63px;
    right: 12.63px;
    width: 14px;
    height: 14px;
    background: transparent;
    border-radius: 50%;
    display: none;
    pointer-events: none;
}

.task-type-card.active .task-type-check {
    display: block;
    background: #524AC9;
}

.task-type-check::after {
    content: '';
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 6px;
    height: 4px;
    border-left: 1.5px solid white;
    border-bottom: 1.5px solid white;
    transform: translate(-50%, -60%) rotate(-45deg);
}

#task_type_hidden {
    display: none !important;
}

/* Keep hidden radio rendered so task-type change events fire */
.task-type-hidden {
    display: none !important;
}

/* File Upload Component - Custom Figma Design */
.gradio-container .file-upload-container {
    display: flex;
    flex-direction: column;
    gap: 20px;
    width: 100%;
    background: #FFFFFF;
    align-items: flex-start;
}

.gradio-container .file-upload-container label {
    display: flex;
    flex-direction: column;
    gap: 3.5px;
}

.gradio-container .file-upload-container label > span:first-child {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 14px;
    line-height: 1.3;
    color: #161616;
}

.gradio-container .file-upload-container label > span:last-child {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 12px;
    line-height: 1.17;
    letter-spacing: 0.32px;
    color: #525252;
}

/* Style the file upload drop area */
.gradio-container input[type="file"] {
    border: 1px dashed #1849D6 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    padding: 24px !important;
    min-height: 120px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.gradio-container input[type="file"]:hover {
    border-color: #524AC9 !important;
    background: #F8F8F8 !important;
}

/* File upload wrapper styling */
.gradio-container .file-upload-container .wrap {
    border: 1px dashed #1849D6 !important;
    border-radius: 8px !important;
    background: #FFFFFF !important;
    padding: 24px !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
    min-height: 120px;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s ease;
    font-size: 0; /* Hide original text */
}

/* Custom upload text */
.gradio-container .file-upload-container .wrap::after {
    content: 'Drag your file(s) or browse';
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 14px;
    line-height: 1.43;
    color: #000000;
    text-align: center;
}

/* Hide the "or" separator */
.gradio-container .file-upload-container .wrap .or {
    display: none;
}

/* Ensure icon is visible */
.gradio-container .file-upload-container .wrap .icon-wrap,
.gradio-container .file-upload-container .wrap svg {
    font-size: initial;
}


.gradio-container .file-upload-container .wrap:hover {
    border-color: #524AC9 !important;
    background: #F8F8F8 !important;
}

/* File upload button and text */
.gradio-container .file-upload-container button {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 14px;
    line-height: 1.43;
    color: #000000;
    background: transparent;
    border: none;
    padding: 0;
    cursor: pointer;
}

.gradio-container .file-upload-container .upload-text {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 14px;
    line-height: 1.43;
    color: #000000;
    text-align: center;
}

#output_section {
    width: 100%;
    max-width: none;
    margin: 0 auto 60px;
    padding: 0;
}

#output_column {
    display: flex;
    flex-direction: column;
    gap: 18px;
    width: 100%;
    padding: 0;
}

#output_column .output-card {
    background: white;
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: 4px;
    padding: 20px;
    box-shadow: 0 14px 24px rgba(8, 19, 51, 0.06);
}

.status-panel {
    border-left: 4px solid var(--carbon-blue-600);
    padding-left: 16px;
}

.status-panel.loading {
    border-color: var(--carbon-blue-600);
}

#log_output textarea {
    background: #0f172a;
    color: #d1d9ff;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    border-radius: 4px;
    border: none;
}

#settings_drawer {
    position: fixed;
    top: 0;
    left: 0;
    width: 400px;
    height: 100vh;
    background: white;
    box-shadow: 0 20px 60px rgba(8, 19, 51, 0.3);
    transform: translateX(-120%);
    transition: transform 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    z-index: 1150;
    display: flex;
    flex-direction: column;
}

#settings_drawer.open {
    transform: translateX(0);
}

#settings_drawer .drawer-header {
    padding: 16px;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    position: relative;
    overflow: hidden;
}

#settings_drawer .drawer-header::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 160px;
    height: 160px;
    opacity: 0.2;
    background: radial-gradient(circle at top right, rgba(15, 98, 254, 0.3), transparent 70%);
    pointer-events: none;
}

#settings_drawer .drawer-header > div {
    position: relative;
    z-index: 1;
}

#drawer_close {
    width: 38px;
    height: 38px;
    border-radius: 8px;
    border: none;
    background: rgba(15, 98, 254, 0.15);
    color: var(--carbon-blue-600);
    font-size: 20px;
    cursor: pointer;
}

#settings_drawer .drawer-body {
    padding: 16px;
    overflow-y: auto;
    flex: 1;
    gap: 16px;
}

#settings_drawer .drawer-footer {
    padding: 16px;
    border-top: 1px solid rgba(0, 0, 0, 0.06);
    background: #f4f4f4;
}

#settings_drawer .drawer-footer button {
    width: 100%;
    border-radius: 0;
    border: none;
    background: var(--carbon-blue-600);
    color: white;
    height: 48px;
    font-weight: 600;
    transition: background 0.11s cubic-bezier(0.2, 0, 0.38, 0.9);
}

#settings_drawer .drawer-footer button:hover {
    background: #0353e9;
}

/* Hover affordances for interactive elements inside step forms */
.step-content [role="button"],
.step-content button,
.step-content .task-type-card,
.step-content .file-upload-container .wrap,
.step-content .form-field .container,
.step-content .form-field label,
.step-content .form-field .svelte-1hfxrpf .secondary-wrap {
    cursor: pointer;
    transition: background-color 0.12s ease, box-shadow 0.12s ease;
}

.step-content [role="button"]:hover,
.step-content button:hover,
.step-content .task-type-card:not(.active):hover,
.step-content .file-upload-container .wrap:hover,
.step-content .form-field .container:hover,
.step-content .form-field .svelte-1hfxrpf .secondary-wrap:hover {
    background-color: #f7f7f7 !important;
    box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.04);
}

#settings_drawer label,
#settings_drawer label span,
#settings_drawer .block-label,
#settings_drawer .block-title {
    background: transparent !important;
    color: #161616 !important;
    border: none !important;
    border-bottom: none !important;
}

@media (max-width: 1024px) {
    .carbon-header__body {
        flex-direction: column;
        align-items: flex-start;
        gap: 16px;
    }

    .header-steps {
        width: 100%;
        flex-wrap: wrap;
        height: auto;
    }

    .header-step {
        flex: 1 1 240px;
        min-height: 60px;
    }

    .carbon-shell {
        padding: 24px 16px 0;
    }

    #config_section,
    #step_info_card {
        padding: 18px;
    }
}

.margin-top-10 {
    margin-top: 10px !improtant;
}

/* ========== Generation Page Styles ========== */

/* Generation Status Banner */
.generation-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 24px;
    border-bottom: 1px solid #EEEEEE;
    min-height: 72px;
    width: 60%;
    max-width: 1200px;
    margin: 0 auto;
}

.generation-banner__content {
    display: flex;
    align-items: center;
    gap: 10.5px;
    flex: 1;
}

.generation-banner__icon {
    width: 48px;
    height: 48px;
    background: #F7F7F7;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}

.generation-banner__icon .iconfont {
    color: #524AC9;
    font-size: 46px;
}

.generation-banner__text {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.generation-banner__title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 16px;
    line-height: 1.2em;
    color: #000000;
    margin: 0;
}

.generation-banner__subtitle {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 12px;
    line-height: 1.2em;
    letter-spacing: 0.0267em;
    color: #525252;
    margin: 0;
    display: flex;
    gap: 4px;
}

.generation-banner__subtitle span.highlight {
    font-weight: 500;
    color: #1E1B4A;
}

.generation-banner.completed .generation-banner__subtitle span.highlight {
    color: #017C4A;
}

.generation-banner__action {
    display: flex;
    align-items: center;
    gap: 10px;
}

.generation-banner__button {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0 16px;
    height: 44px;
    border-radius: 8px;
    border: 1px solid;
    background: #FFFFFF;
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 16px;
    line-height: 1em;
    cursor: pointer;
    transition: all 0.2s ease;
}

.generation-banner__button.download {
    border-color: #524AC9;
    color: #524AC9;
}

.generation-banner__button.download .iconfont {
    color: #524AC9;
}

.generation-banner__button.download:hover {
    background: rgba(82, 74, 201, 0.08);
}

.generation-banner__button.cancel {
    border-color: #858585;
    color: #858585;
}

.generation-banner__button.cancel:hover {
    background: rgba(133, 133, 133, 0.08);
}

/* Progress Cards Container */
#progress_cards_container {
    display: flex;
    flex-direction: column;
    gap: 14px;
    padding: 24px;
    margin: 0 auto;
    width: 60%;
    max-width: 1200px;
}

/* Progress Card */
.progress-card {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 20px 16px;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.progress-card.default {
    background: rgba(36, 161, 72, 0);
}

.progress-card.in-progress {
    background: rgba(82, 74, 201, 0.05);
}

.progress-card.completed {
    background: rgba(3, 124, 74, 0.02);
}

.progress-card__icon {
    width: 18px;
    height: 18px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* Icon colors for different states */
.progress-card.default .progress-card__icon .iconfont {
    color: #E0E0E0;
    font-size: 18px;
}

.progress-card.in-progress .progress-card__icon .iconfont {
    color: #524AC9;
    font-size: 18px;
    animation: rotate 1.5s linear infinite;
}

.progress-card.completed .progress-card__icon .iconfont {
    color: #017C4A;
    font-size: 18px;
}

/* Rotation animation for in-progress icon */
@keyframes rotate {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

.progress-card__content {
    display: flex;
    flex-direction: column;
    gap: 3.5px;
    flex: 1;
}

.progress-card__text {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 14px;
    line-height: 1.2em;
    letter-spacing: 0.0114em;
    margin: 0;
}

.progress-card.default .progress-card__text {
    color: #858585;
}

.progress-card.in-progress .progress-card__text {
    color: #524AC9;
}

.progress-card.completed .progress-card__text {
    color: #037C4A;
}

.progress-card__badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 3px 9px;
    border-radius: 4px;
    border: 1px solid;
    align-self: flex-start;
}

.progress-card.completed .progress-card__badge {
    background: rgba(3, 124, 74, 0.02);
    border-color: rgba(3, 124, 74, 0.3);
}

.progress-card.in-progress .progress-card__badge {
    background: rgba(82, 74, 201, 0.05);
    border-color: rgba(82, 74, 201, 0.3);
}

.progress-card__badge-text {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 10.5px;
    line-height: 1.33em;
    margin: 0;
}

.progress-card.completed .progress-card__badge-text {
    color: #037C4A;
}

.progress-card.in-progress .progress-card__badge-text {
    color: #524AC9;
}

/* Output Files Section */
#output_files_section {
    display: flex;
    flex-direction: column;
    gap: 28px;
    padding: 28px;
    border: 1px dashed #E5E5E5;
    border-radius: 4px;
    margin: 20px auto 0;
    width: 60%;
    max-width: 1200px;
}

.output-files__title {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 500;
    font-size: 16px;
    line-height: 1.09em;
    color: #000000;
    margin: 0;
}

.output-files__list {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.output-file-item {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.output-file-item__label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 14px;
    line-height: 1.25em;
    color: #000000;
    margin: 0;
}

.output-file-item__path {
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 400;
    font-size: 14px;
    line-height: 1.25em;
    color: #858585;
    margin: 0;
}

/* Loading spinner for in-progress state */
@keyframes spin {
    0% {
        transform: rotate(0deg);
    }
    100% {
        transform: rotate(360deg);
    }
}

.spinner {
    width: 18px;
    height: 18px;
    border: 2px solid rgba(82, 74, 201, 0.2);
    border-top-color: #524AC9;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Hidden download file component - keep in DOM but visually hidden */
#hidden_download_file {
    position: absolute !important;
    width: 1px !important;
    height: 1px !important;
    padding: 0 !important;
    margin: -1px !important;
    overflow: hidden !important;
    clip: rect(0, 0, 0, 0) !important;
    white-space: nowrap !important;
    border: 0 !important;
}
"""

CUSTOM_HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Sans+3:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="//at.alicdn.com/t/c/font_5067036_7p52hcxtupi.css" />
<script src="//at.alicdn.com/t/c/font_5067036_7p52hcxtupi.js"></script>
<script>
(function() {
    const debugLog = (...args) => console.debug('[SDG Carbon]', ...args);

    // Step Wizard Management
    let currentStep = 1;
    let totalSteps = 3; // 3 steps: Basic Settings, Task Configuration, Advanced Settings

    window.__sdgSetTotalSteps = (steps) => {
        totalSteps = steps;
        console.debug('[SDG Wizard] Total steps set to:', steps);

        // Trigger UI update to reflect new total steps
        const root = getRoot();
        if (root && window.__sdgGetCurrentStep) {
            const current = window.__sdgGetCurrentStep();
            // Force update of step indicators and buttons
            setTimeout(() => {
                if (current <= steps) {
                    window.__sdgChangeStep(current, true);
                } else {
                    window.__sdgChangeStep(steps, true);
                }
            }, 50);
        }
    };

    window.__sdgGetCurrentStep = () => currentStep;

    window.__sdgChangeStep = (step, allowForward = false) => {
        if (step < 1 || step > totalSteps) return;

        // Only allow backward navigation unless explicitly allowed
        if (!allowForward && step > currentStep) {
            console.debug('[SDG Wizard] Cannot jump forward from step', currentStep, 'to', step);
            return;
        }

        currentStep = step;

        // Update step indicators (gradio cards)
        const root = getRoot();
        if (!root) return;

        // Update step indicators in the bottom nav bar
        const navItems = root.querySelectorAll('.carbon-header__bottom .header-nav__item');
        navItems.forEach((item, index) => {
            const stepNum = index + 1;
            item.classList.remove('active', 'completed');

            if (stepNum === currentStep) {
                item.classList.add('active');
            } else if (stepNum < currentStep) {
                item.classList.add('completed');
            }

            // Update clickability: can click if step <= currentStep
            if (stepNum <= currentStep) {
                item.classList.add('clickable');
            } else {
                item.classList.remove('clickable');
            }

            // Hide step 3 if totalSteps is 2
            if (stepNum === 3 && totalSteps === 2) {
                item.style.display = 'none';
            } else {
                item.style.display = 'flex';
            }
        });

        const stepContents = root.querySelectorAll('.step-content');

        // Update content visibility
        stepContents.forEach((content, index) => {
            content.classList.remove('active');
            if (index === currentStep - 1) {
                content.classList.add('active');
            }
        });

        // Update button visibility
        const prevBtn = root.querySelector('#btn_previous');
        const nextBtn = root.querySelector('#btn_next');
        const submitBtn = root.querySelector('#btn_submit');

        if (prevBtn) {
            if (currentStep === 1) {
                prevBtn.style.opacity = '0';
                prevBtn.style.pointerEvents = 'none';
            } else {
                prevBtn.style.opacity = '1';
                prevBtn.style.pointerEvents = 'auto';
            }
        }
        if (nextBtn) nextBtn.style.display = currentStep === totalSteps ? 'none' : 'block';
        if (submitBtn) submitBtn.style.display = currentStep === totalSteps ? 'block' : 'none';

        // Update nav states between Configuration / Generate tabs
        syncHeaderNavigation();
    };

    window.__sdgNextStep = () => {
        if (currentStep < totalSteps) {
            window.__sdgChangeStep(currentStep + 1, true);
        }
    };

    window.__sdgPreviousStep = () => {
        if (currentStep > 1) {
            window.__sdgChangeStep(currentStep - 1);
        }
    };

    window.__sdgScrollToOutput = () => {
        setTimeout(() => {
            const root = getRoot();
            if (!root) return;

            const outputColumn = root.querySelector('#output_column');
            if (outputColumn) {
                outputColumn.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 300);
    };

    window.__sdgShowOutput = () => {
        const root = getRoot();
        if (!root) return;

        // Remove wizard mode styling
        const mainLayout = root.querySelector('#main_layout');
        if (mainLayout) {
            mainLayout.classList.remove('wizard-mode');
        }

        // Show output section (Gradio visibility)
        // We need to trigger Gradio's visibility update
        // The actual visibility is controlled by the generate_btn output

        // Scroll to output after a short delay
        setTimeout(() => {
            const outputSection = root.querySelector('#output_section');
            if (outputSection) {
                outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 300);

        // 安全设置 generated 标记
        const datasetHolder = root.dataset ? root : (root.host || document.querySelector('gradio-app'));
        if (datasetHolder && datasetHolder.dataset) {
            datasetHolder.dataset.generated = 'true';
        }

        syncHeaderNavigation();
        setNavMode('generate');
    };

    window.__sdgDownloadData = () => {
        // Search in the entire document, not just the root
        const fileComponent = document.querySelector('#hidden_download_file');
        if (!fileComponent) {
            console.warn('Download file component not found');
            alert('Download component not found. Please try again.');
            return;
        }

        // Find the download button within the gr.File component
        // Gradio's File component has a download button that we can click
        const downloadButton = fileComponent.querySelector('a[download], button[download]');
        if (downloadButton) {
            console.log('Triggering download via Gradio File component');
            downloadButton.click();
        } else {
            // Fallback: look for any clickable element in the file component
            const anyButton = fileComponent.querySelector('button, a');
            if (anyButton) {
                console.log('Triggering download via fallback button');
                anyButton.click();
            } else {
                console.warn('No download button found in file component');
                alert('No file available for download. Please check if the generation completed successfully.');
            }
        }
    };

    window.__sdgInitWizardMode = () => {
        const root = getRoot();
        if (!root) return;

        const mainLayout = root.querySelector('#main_layout');
        if (mainLayout) {
            mainLayout.classList.add('wizard-mode');
            mainLayout.classList.remove('split-mode');
        }

        setNavMode('configuration');
    };

    const getRoot = () => {
        if (typeof window.gradioApp === 'function') {
            const appRoot = window.gradioApp();
            if (appRoot) {
                return appRoot;
            }
        }
        return document.querySelector('gradio-app')?.shadowRoot || document;
    };

    const select = (selector) => {
        const root = getRoot();
        if (!root) {
            debugLog('root not ready when selecting', selector);
            return null;
        }
        const found = root.querySelector(selector);
        if (found) {
            return found;
        }
        if (selector.startsWith('#')) {
            const fuzzy = root.querySelector(`[id^="${selector.slice(1)}"]`);
            if (fuzzy) {
                debugLog('fuzzy match for selector', selector, '->', fuzzy.id);
                return fuzzy;
            }
        }
        debugLog('selector not found', selector);
        return null;
    };

    const getElements = () => ({
        drawer: select('#settings_drawer'),
        overlay: select('#drawer_overlay'),
        toggle: select('#settings_toggle'),
        close: select('#drawer_close'),
        headerConfig: select('#header_config_btn'),
        headerGenerate: select('#header_generate_btn'),
        navConfiguration: selectAll('[data-nav="configuration"]'),
        navGenerate: selectAll('[data-nav="generate"]'),
        generateBtn: select('#btn_submit button, #btn_submit'),
        configSection: select('#config_section'),
        outputSection: select('#output_section')
    });

    const selectAll = (selector) => {
        const root = getRoot();
        if (!root) return [];
        const list = Array.from(root.querySelectorAll(selector));
        if (list.length === 0 && selector.startsWith('[data-nav')) {
            debugLog('no nav elements found for', selector);
        }
        return list;
    };

    const lockScroll = () => {
        if (document.body.dataset.sdgScrollLocked === 'true') {
            return;
        }
        document.body.dataset.sdgScrollLocked = 'true';
        document.body.style.overflow = 'hidden';
        document.body.style.touchAction = 'none';
    };

    const unlockScroll = () => {
        if (document.body.dataset.sdgScrollLocked !== 'true') {
            return;
        }
        delete document.body.dataset.sdgScrollLocked;
        document.body.style.overflow = '';
        document.body.style.touchAction = '';
    };

    const openDrawer = () => {
        const { drawer, overlay } = getElements();
        if (!drawer) {
            console.warn('[SDG Drawer] drawer not found when trying to open');
            return;
        }
        drawer.classList.add('open');
        overlay?.classList.add('active');
        debugLog('drawer opened');
        lockScroll();
    };

    const closeDrawer = () => {
        const { drawer, overlay } = getElements();
        if (!drawer) {
            return;
        }
        drawer.classList.remove('open');
        overlay?.classList.remove('active');
        debugLog('drawer closed');
        unlockScroll();
    };

    const bindClick = (element, handler, name) => {
        if (!element) {
            debugLog('bindClick skipped for missing element', name);
            return;
        }
        if (element.dataset.sdgBound === 'true') {
            return;
        }
        element.addEventListener('click', handler);
        element.dataset.sdgBound = 'true';
        debugLog('bound click handler for', name);
    };

    const ensureBindings = () => {
        const { toggle, close, overlay, headerConfig, headerGenerate, navConfiguration, navGenerate, generateBtn } = getElements();
        bindClick(toggle, openDrawer, 'settings_toggle');
        bindClick(close, closeDrawer, 'drawer_close');
        bindClick(overlay, closeDrawer, 'drawer_overlay');
        bindClick(headerConfig, openDrawer, 'header_config_btn');
        bindClick(headerGenerate, () => {
            if (generateBtn && !headerGenerate.classList.contains('disabled')) {
                generateBtn.click();
            }
        }, 'header_generate_btn');

        (navConfiguration || []).forEach((btn, idx) => {
            bindClick(btn, () => {
                setNavMode('configuration');
            }, `nav_configuration_${idx}`);
        });

        (navGenerate || []).forEach((btn, idx) => {
            bindClick(btn, () => {
                if (btn.classList.contains('disabled')) {
                    return;
                }
                // 只切换视图，不重新触发生成
                // 用户已经生成过数据，点击这里只是查看结果
                setNavMode('generate');
            }, `nav_generate_${idx}`);
        });

        // Bind step item clicks
        const root = getRoot();
        if (root) {
            // Bind bottom navigation step clicks
            const navItems = root.querySelectorAll('.carbon-header__bottom .header-nav__item');
            navItems.forEach((item, index) => {
                const stepNum = index + 1;
                if (item.dataset.sdgStepBound !== 'true') {
                    item.addEventListener('click', (e) => {
                        // Only navigate if clickable (step has been visited or is current/previous)
                        if (item.classList.contains('clickable')) {
                            window.__sdgChangeStep(stepNum);
                        } else {
                            // Provide visual feedback that forward navigation is not allowed
                            e.stopPropagation();
                            debugLog('[SDG Wizard] Cannot navigate to step', stepNum, '- step not yet visited');
                        }
                    });
                    item.dataset.sdgStepBound = 'true';
                    debugLog('bound click handler for nav step', stepNum);
                }
            });
        }

        // Check if all required fields are filled
        validateRequiredFields();

        syncHeaderNavigation();
        return { toggle, close, overlay };
    };

    const setNavMode = (mode) => {
        const root = getRoot();
        if (!root) return;

        // 安全设置 navMode：如果 root 是 ShadowRoot，使用 host 元素
        const datasetHolder = root.dataset ? root : (root.host || document.querySelector('gradio-app'));
        if (datasetHolder && datasetHolder.dataset) {
            datasetHolder.dataset.navMode = mode;
        }

        const isConfiguration = mode === 'configuration';
        const navConfiguration = root.querySelectorAll('[data-nav="configuration"]');
        const navGenerate = root.querySelectorAll('[data-nav="generate"]');
        const mainLayout = root.querySelector('#config_section');
        const outputSection = root.querySelector('#output_section');
        const bottomNav = root.querySelector('.carbon-header__bottom');

        navConfiguration.forEach((btn) => {
            btn.classList.toggle('active', isConfiguration);
        });
        navGenerate.forEach((btn) => {
            btn.classList.toggle('active', !isConfiguration);
        });

        if (mainLayout) {
            mainLayout.classList.toggle('hidden', !isConfiguration);
            mainLayout.style.display = isConfiguration ? 'flex' : 'none';
        }

        if (outputSection) {
            outputSection.style.display = isConfiguration ? 'none' : 'flex';
        }

        // 控制底部步骤导航的显示/隐藏
        // Configuration 模式：显示步骤导航
        // Generate Dataset 模式：隐藏步骤导航
        if (bottomNav) {
            bottomNav.style.display = isConfiguration ? 'block' : 'none';
        }
    };

    const syncHeaderNavigation = () => {
        const root = getRoot();
        if (!root) return;

        // 安全获取 dataset：如果 root 是 ShadowRoot，使用 host 元素
        const datasetHolder = root.dataset ? root : (root.host || document.querySelector('gradio-app'));
        if (!datasetHolder || !datasetHolder.dataset) {
            // 如果无法获取 dataset，默认禁用 Generate Dataset tab
            const navGenerate = root.querySelectorAll('[data-nav="generate"]');
            navGenerate.forEach((btn) => {
                btn.classList.add('disabled');
            });
            return;
        }

        const navGenerate = root.querySelectorAll('[data-nav="generate"]');
        const hasGenerated = datasetHolder.dataset.generated === 'true';
        navGenerate.forEach((btn) => {
            if (hasGenerated) {
                btn.classList.remove('disabled');
            } else {
                btn.classList.add('disabled');
            }
        });
    };

    // Validate required fields and update Generate button
    const validateRequiredFields = () => {
        const root = getRoot();
        if (!root) return;

        const generateBtn = root.querySelector('#header_generate_btn');
        if (!generateBtn) return;

        // Required fields
        const taskName = root.querySelector('input[placeholder*="gsm8k"]');
        const domain = root.querySelector('input[placeholder*="mathematics"]');
        const taskInstruction = root.querySelector('textarea[placeholder*="What kind of data"]');

        // Check if translator model path is required (when Arabic is selected)
        const languageDropdown = root.querySelector('#component-') ||
                                Array.from(root.querySelectorAll('select, input[role="listbox"]')).find(el => {
                                    const parent = el.closest('.form-field');
                                    return parent && parent.textContent.includes('Dataset Language');
                                });
        const translatorGroup = root.querySelector('#translator_model_group');
        const translatorInput = translatorGroup?.querySelector('input[type="text"]');

        let translatorRequired = false;
        if (languageDropdown && translatorGroup) {
            const isArabic = languageDropdown.value?.toLowerCase() === 'arabic' ||
                           languageDropdown.textContent?.toLowerCase().includes('arabic');
            const groupVisible = translatorGroup.style.display !== 'none' &&
                                !translatorGroup.classList.contains('hidden');
            translatorRequired = isArabic && groupVisible && !translatorInput?.value.trim();
        }

        const allFilled = taskName?.value.trim() &&
                         domain?.value.trim() &&
                         taskInstruction?.value.trim() &&
                         !translatorRequired;

        if (allFilled) {
            generateBtn.classList.remove('disabled');
            generateBtn.style.backgroundColor = '#524AC9';
            generateBtn.style.cursor = 'pointer';
            generateBtn.style.opacity = '1';
            generateBtn.style.border = 'none';
        } else {
            generateBtn.classList.add('disabled');
            generateBtn.style.backgroundColor = 'rgba(255, 255, 255, 0.12)';
            generateBtn.style.cursor = 'not-allowed';
            generateBtn.style.opacity = '1';
            generateBtn.style.border = '1px solid transparent';

            // Also update the text color
            const btnText = generateBtn.querySelector('span:last-child');
            if (btnText) {
                btnText.style.color = allFilled ? 'white' : 'rgba(255, 255, 255, 0.4)';
            }
        }
    };

    const pruneEmptyGenerationWraps = () => {
        const root = getRoot();
        if (!root) {
            return;
        }
        const wraps = root.querySelectorAll('.wrap.center.full.svelte-btia7y.generating');
        wraps.forEach((wrap) => {
            const hasVisibleChildren = Array.from(wrap.children || []).some((child) => {
                const tag = child.tagName ? child.tagName.toLowerCase() : '';
                return tag !== 'style' && tag !== 'script';
            });
            const hasText = (wrap.textContent || '').trim().length > 0;
            if (!hasVisibleChildren && !hasText) {
                wrap.remove();
            }
        });
    };

    const fixStatusPanelWidth = () => {
        const root = getRoot();
        if (!root) {
            return;
        }
        // 修复所有 .md span 元素的宽度问题
        const mdSpans = root.querySelectorAll('#output_column span.md, #output_column .prose.md');
        mdSpans.forEach((span) => {
            span.style.display = 'block';
            span.style.width = '100%';
            span.style.maxWidth = '100%';
            span.style.boxSizing = 'border-box';
        });

        // 修复 status-panel-content 的宽度
        const statusContents = root.querySelectorAll('.status-panel-content');
        statusContents.forEach((content) => {
            content.style.width = '100%';
            content.style.display = 'flex';
            content.style.boxSizing = 'border-box';
        });

        // 修复 status-panel 的宽度
        const statusPanels = root.querySelectorAll('.status-panel');
        statusPanels.forEach((panel) => {
            panel.style.width = '100%';
            panel.style.maxWidth = '100%';
            panel.style.boxSizing = 'border-box';
        });
    };

    const ensureKeyBinding = () => {
        if (window.__sdgKeyBound) {
            return;
        }
        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') {
                closeDrawer();
            }
        });
        window.__sdgKeyBound = true;
        debugLog('Escape key handler attached');
    };

    const observer = new MutationObserver((mutations) => {
        ensureBindings();
        pruneEmptyGenerationWraps();
        fixStatusPanelWidth();
        validateRequiredFields();
        syncHeaderNavigation();
        setupTaskTypeCards();
        handleDropdownStates();
    });

    const startObserving = () => {
        const host = document.querySelector('gradio-app') || document.body;
        if (!host) {
            debugLog('host not found, retry in 500ms');
            setTimeout(startObserving, 500);
            return;
        }
        observer.observe(host, { subtree: true, childList: true });
        debugLog('observer attached to host');
        ensureBindings();
        pruneEmptyGenerationWraps();
        fixStatusPanelWidth();
        handleDropdownStates();

        setNavMode('configuration');
    };

    ensureKeyBinding();

    // Mark optional fields (those with "Optional" in label text)
    const markOptionalFields = () => {
        const root = getRoot();
        if (!root) return;

        // Find all labels and mark those containing "Optional" text
        const labels = root.querySelectorAll('label span[data-testid="block-info"]');
        labels.forEach(label => {
            if (label.textContent.includes('Optional')) {
                label.classList.add('optional-label');
            }
        });
    };

    // Setup Task Type card selection
    const setupTaskTypeCards = () => {
        const root = getRoot();
        if (!root) return;

        const cards = root.querySelectorAll('.task-type-card');
        const hiddenRadio = root.querySelector('#task_type_hidden');

        if (!cards.length || !hiddenRadio) return;

        // Prevent duplicate binding
        cards.forEach((card, index) => {
            if (card.dataset.sdgBound === 'true') return;

            card.addEventListener('click', () => {
                // Remove active class from all cards
                cards.forEach(c => c.classList.remove('active'));

                // Add active class to clicked card
                card.classList.add('active');

                // Update hidden radio value
                const value = card.dataset.value;
                const radioInputs = hiddenRadio.querySelectorAll('input[type="radio"]');
                radioInputs.forEach(input => {
                    if (input.value === value) {
                        input.checked = true;
                        // Trigger change event
                        input.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                });

                debugLog('Task type changed to:', value);
            });

            card.dataset.sdgBound = 'true';
        });

        debugLog('Task type cards initialized');
    };

    // Handle dropdown open/close states for arrow rotation
    const handleDropdownStates = () => {
        const root = getRoot();
        if (!root) return;

        // Find all dropdown wrappers
        const dropdownWrappers = root.querySelectorAll('.wrap:has(> input[role="listbox"])');

        dropdownWrappers.forEach(wrapper => {
            // Skip if already bound
            if (wrapper.dataset.dropdownBound === 'true') return;

            const input = wrapper.querySelector('input[role="listbox"]');
            if (!input) return;

            // Add click listener to toggle state
            const toggleDropdownState = () => {
                // Check if options are visible (dropdown is open)
                const optionsWrap = wrapper.querySelector('.options-wrap, [class*="options"]');
                const isOpen = optionsWrap && optionsWrap.offsetParent !== null;

                if (isOpen) {
                    wrapper.classList.add('dropdown-open');
                } else {
                    wrapper.classList.remove('dropdown-open');
                }
            };

            // Listen for clicks on the input
            input.addEventListener('click', () => {
                setTimeout(toggleDropdownState, 50);
            });

            // Listen for clicks outside to close
            document.addEventListener('click', (e) => {
                if (!wrapper.contains(e.target)) {
                    wrapper.classList.remove('dropdown-open');
                }
            });

            wrapper.dataset.dropdownBound = 'true';
        });
    };

    // Global function for task type selection (called from onclick)
    window.__selectTaskType = (value, clickedCard) => {
        // Prevent selection of disabled cards
        if (clickedCard.classList.contains('disabled')) {
            return;
        }

        // Get root - could be shadow DOM or regular DOM
        const root = clickedCard.getRootNode();

        // Remove active class from all cards and switch icons back to normal
        const cards = root.querySelectorAll('.task-type-card');
        cards.forEach(c => {
            c.classList.remove('active');

            // Switch icon back to normal state
            const icon = c.querySelector('.task-type-icon');
            if (icon) {
                const dataValue = c.dataset.value;
                icon.classList.remove(`icon-synthetic-${dataValue}-active`);
                icon.classList.add(`icon-synthetic-${dataValue}`);
            }
        });

        // Add active class to clicked card and switch to active icon
        clickedCard.classList.add('active');

        // Switch icon to active state
        const activeIcon = clickedCard.querySelector('.task-type-icon');
        if (activeIcon) {
            activeIcon.classList.remove(`icon-synthetic-${value}`);
            activeIcon.classList.add(`icon-synthetic-${value}-active`);
        }

        // Update hidden radio value
        const hiddenRadio = root.querySelector('#task_type_hidden') ||
                           root.querySelector('[id*="task_type_hidden"]');
        if (hiddenRadio) {
            const radioInputs = hiddenRadio.querySelectorAll('input[type="radio"]');
            radioInputs.forEach(input => {
                if (input.value === value) {
                    input.checked = true;
                    // Trigger change event
                    input.dispatchEvent(new Event('change', { bubbles: true }));
                }
            });
        }

        console.log('[SDG Task Type] Selected:', value);
    };

    // Initialize wizard steps and layout
    setTimeout(() => {
        window.__sdgChangeStep && window.__sdgChangeStep(1, true);
        window.__sdgInitWizardMode && window.__sdgInitWizardMode();

        // Initialize step count based on default task type
        window.__sdgSetTotalSteps && window.__sdgSetTotalSteps(3);

        // Mark optional fields after initialization
        markOptionalFields();

        // Setup task type cards
        setupTaskTypeCards();

        // Initialize default task type icon to active state (Local is default)
        const root = getRoot();
        if (root) {
            const defaultCard = root.querySelector('.task-type-card.active');
            if (defaultCard) {
                const defaultIcon = defaultCard.querySelector('.task-type-icon');
                const dataValue = defaultCard.dataset.value;
                if (defaultIcon && dataValue) {
                    defaultIcon.classList.remove(`icon-synthetic-${dataValue}`);
                    defaultIcon.classList.add(`icon-synthetic-${dataValue}-active`);
                }
            }
        }
    }, 500);

    startObserving();
    pruneEmptyGenerationWraps();
    fixStatusPanelWidth();

    window.__sdgDrawerDebug = () => {
        const elements = getElements();
        debugLog('debug snapshot', elements);
        return elements;
    };

    window.__sdgForceBindings = () => {
        const elements = ensureBindings();
        debugLog('force bindings invoked', elements);
        return elements;
    };

    window.__sdgToggleDrawer = (open = true) => {
        if (open) {
            openDrawer();
        } else {
            closeDrawer();
        }
    };

    window.__sdgLogGradioApp = () => {
        const root = getRoot();
        debugLog('manual log gradio root', root);
        debugLog('window.gradioApp()', typeof window.gradioApp === 'function' ? window.gradioApp() : undefined);
        return root;
    };
})();
</script>
"""


# Generation Steps Definition
GENERATION_STEPS = [
    "Generating initial dataset...",
    "Evaluating initial dataset...",
    "Rewriting samples based on difficulty...",
    "Re-evaluating rewritten samples...",
    "Categorizing results...",
    "Translating datasets to {language}...",
    "Saving datasets...",
    "Cleaning up models..."
]


def create_error_banner_html(error_message: str):
    """
    Generate HTML for error status banner

    Args:
        error_message: Error message to display

    Returns:
        HTML string for error banner
    """
    return f'''
    <div class="generation-banner" style="border-bottom: 1px solid #DA1E28;">
        <div class="generation-banner__content">
            <div class="generation-banner__icon" style="background: rgba(218, 30, 40, 0.1);">❌</div>
            <div class="generation-banner__text">
                <p class="generation-banner__title" style="color: #DA1E28;">Generation Failed</p>
                <p class="generation-banner__subtitle" style="color: #525252;">
                    An error occurred during the generation process
                </p>
            </div>
        </div>
        <div class="generation-banner__action">
            <button class="generation-banner__button cancel" onclick="window.location.reload()" style="border-color: #DA1E28; color: #DA1E28;">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                    <path d="M14 4.5L4 14.5M14 14.5L4 4.5" stroke="#DA1E28" stroke-width="1.5" stroke-linecap="round"/>
                </svg>
                Reset
            </button>
        </div>
    </div>
    '''


def create_error_message_html(error_message: str):
    """
    Generate HTML for error message display

    Args:
        error_message: Error message to display

    Returns:
        HTML string for error message
    """
    return f'''
    <div style="padding: 32px; background: rgba(218, 30, 40, 0.05); border: 1px solid rgba(218, 30, 40, 0.2); border-radius: 8px; margin-top: 24px;">
        <div style="display: flex; align-items: flex-start; gap: 16px;">
            <div style="flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%; background: #DA1E28; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 16px;">!</div>
            <div style="flex: 1;">
                <h3 style="margin: 0 0 12px 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 18px; font-weight: 600; color: #DA1E28;">Error Details</h3>
                <p style="margin: 0; font-family: 'IBM Plex Sans', monospace; font-size: 14px; line-height: 1.6; color: #161616; background: white; padding: 16px; border-radius: 4px; border: 1px solid rgba(218, 30, 40, 0.2); white-space: pre-wrap; word-break: break-word;">{error_message}</p>
            </div>
        </div>
    </div>
    '''


def create_status_banner_html(status: str, num_samples: int, model_name: str, is_complete: bool = False, download_file_path: str = None):
    """
    Generate HTML for the generation status banner

    Args:
        status: "in-progress" or "completed"
        num_samples: Number of data samples being generated
        model_name: Name of the LLM model being used
        is_complete: Whether generation is complete
        download_file_path: Path to file for download (optional)

    Returns:
        HTML string for the banner
    """
    if is_complete:
        title = "Generation Complete!"
        subtitle_text = "Generated"
        icon = '<i class="iconfont icon-synthetic-a-NavGeneratingIcon"></i>'  # Icon for completed
        button_html = '''
            <button class="generation-banner__button download" onclick="window.__sdgDownloadData()">
                <i class="iconfont icon-synthetic-a-Property1Download6-active"></i>
                Download Data
            </button>
        '''
    else:
        title = "Data Generation in Progress..."
        subtitle_text = "Generating"
        icon = '<i class="iconfont icon-synthetic-a-NavGeneratingIcon"></i>'  # Icon for in progress
        button_html = ''  # No button during generation

    return f'''
    <div class="generation-banner {'completed' if is_complete else 'in-progress'}" data-download-path="{download_file_path or ''}">
        <div class="generation-banner__content">
            <div class="generation-banner__icon">{icon}</div>
            <div class="generation-banner__text">
                <p class="generation-banner__title">{title}</p>
                <p class="generation-banner__subtitle">
                    {subtitle_text}
                    <span class="highlight">{num_samples}</span>
                    data samples using
                    <span class="highlight">{model_name}</span>
                </p>
            </div>
        </div>
        <div class="generation-banner__action">
            {button_html}
        </div>
    </div>
    '''


def create_progress_cards_html(steps: list, current_step_index: int, current_step_usage: str = None, completed_step_stats: dict = None, step_dataset_info: dict = None):
    """
    Generate HTML for all progress step cards with optional usage information

    Args:
        steps: List of step descriptions
        current_step_index: Index of current step (0-based). Use -1 for not started, len(steps) for all completed
        current_step_usage: Optional usage counter information for the current step (token usage, time estimates)
        completed_step_stats: Dict mapping step index to completion stats (token, time)
        step_dataset_info: Dict mapping step index to dataset list information (for web task)

    Returns:
        HTML string for all progress cards
    """
    cards_html = []
    completed_step_stats = completed_step_stats or {}
    step_dataset_info = step_dataset_info or {}

    for idx, step_text in enumerate(steps):
        usage_detail_html = ''

        if idx < current_step_index:
            # Completed
            status_class = "completed"
            icon_svg = '<i class="iconfont icon-synthetic-a-GenerationStepComplete"></i>'
            badge_html = '<div class="progress-card__badge"><span class="progress-card__badge-text">Completed</span></div>'

            # Show completion stats if available
            if idx in completed_step_stats:
                stats = completed_step_stats[idx]
                usage_detail_html += f'''
                <div style="margin-top: 8px; padding: 8px; background: rgba(3, 124, 74, 0.05); border-radius: 4px; border: 1px solid rgba(3, 124, 74, 0.1);">
                    <p style="margin: 0; font-size: 11px; line-height: 1.4; color: #037C4A; font-family: 'IBM Plex Sans', monospace; white-space: pre-wrap;">{stats}</p>
                </div>
                '''

            # Show dataset info if available (for completed steps)
            if idx in step_dataset_info:
                dataset_info = step_dataset_info[idx]
                usage_detail_html += f'''
                <div style="margin-top: 8px; padding: 8px; background: rgba(3, 124, 74, 0.05); border-radius: 4px; border: 1px solid rgba(3, 124, 74, 0.1);">
                    <p style="margin: 0; font-size: 11px; line-height: 1.4; color: #037C4A; font-family: 'IBM Plex Sans', monospace; white-space: pre-wrap;">{dataset_info}</p>
                </div>
                '''
        elif idx == current_step_index:
            # In Progress
            status_class = "in-progress"
            icon_svg = '<i class="iconfont icon-synthetic-a-GenerationStepProgressing"></i>'
            badge_html = '<div class="progress-card__badge"><span class="progress-card__badge-text">In Progress</span></div>'

            # Add usage info for current step if available
            if current_step_usage:
                usage_detail_html += f'''
                <div style="margin-top: 8px; padding: 8px; background: rgba(82, 74, 201, 0.05); border-radius: 4px; border: 1px solid rgba(82, 74, 201, 0.1);">
                    <p style="margin: 0; font-size: 11px; line-height: 1.4; color: #524AC9; font-family: 'IBM Plex Sans', monospace; white-space: pre-wrap;">{current_step_usage}</p>
                </div>
                '''

            # Show dataset info if available (for current step)
            if idx in step_dataset_info:
                dataset_info = step_dataset_info[idx]
                usage_detail_html += f'''
                <div style="margin-top: 8px; padding: 8px; background: rgba(82, 74, 201, 0.05); border-radius: 4px; border: 1px solid rgba(82, 74, 201, 0.1);">
                    <p style="margin: 0; font-size: 11px; line-height: 1.4; color: #524AC9; font-family: 'IBM Plex Sans', monospace; white-space: pre-wrap;">{dataset_info}</p>
                </div>
                '''
        else:
            # Default (waiting)
            status_class = "default"
            icon_svg = '<i class="iconfont icon-synthetic-a-GenerationStepDefault"></i>'
            badge_html = ''

        card_html = f'''
        <div class="progress-card {status_class}">
            <div class="progress-card__icon">{icon_svg}</div>
            <div class="progress-card__content">
                <p class="progress-card__text">{step_text}</p>
                {usage_detail_html}
            </div>
            {badge_html}
        </div>
        '''
        cards_html.append(card_html)

    return '<div id="progress_cards_container">' + '\n'.join(cards_html) + '</div>'


def create_output_files_html(output_dir: str, task_name: str, export_format: str = "jsonl"):
    """
    Generate HTML for output files section - only shows files that actually exist

    Args:
        output_dir: Output directory path
        task_name: Name of the task
        export_format: File format (jsonl, json, etc.)

    Returns:
        HTML string for output files section
    """
    base_path = f"{output_dir}/{task_name}_final"

    # Check which files actually exist
    files_info = []

    solved_file = f"{base_path}_solved.{export_format}"
    if os.path.exists(solved_file):
        files_info.append({
            "label": "Solved samples:",
            "path": solved_file
        })

    learnable_file = f"{base_path}_learnable.{export_format}"
    if os.path.exists(learnable_file):
        files_info.append({
            "label": "Learnable samples:",
            "path": learnable_file
        })

    unsolved_file = f"{base_path}_unsolved.{export_format}"
    if os.path.exists(unsolved_file):
        files_info.append({
            "label": "Unsolved samples:",
            "path": unsolved_file
        })

    # Generate HTML for existing files only
    if not files_info:
        return '''
        <div id="output_files_section">
            <h3 class="output-files__title">Output Files Created:</h3>
            <p style="color: #525252; font-size: 14px;">No output files were created (all categories were empty).</p>
        </div>
        '''

    files_html = []
    for file_info in files_info:
        files_html.append(f'''
            <div class="output-file-item">
                <p class="output-file-item__label">{file_info["label"]}</p>
                <p class="output-file-item__path">{file_info["path"]}</p>
            </div>
        ''')

    return f'''
    <div id="output_files_section">
        <h3 class="output-files__title">Output Files Created:</h3>
        <div class="output-files__list">
            {''.join(files_html)}
        </div>
    </div>
    '''


def create_config_dict(
    # Essential fields
    task_type: str,
    task_instruction: str,
    input_instruction: str,
    output_instruction: str,
    num_samples: int,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    # Basic settings
    task_name: str,
    domain: str,
    output_dir: str,
    cuda_device: str,
    hf_token: str,
    language: str,
    # Model paths
    base_model_path: str,
    semantic_model_path: str,
    # Local task specific
    documents: Optional[List] = None,
    # Web task specific
    dataset_score_threshold: int = 30,
    # Answer extraction
    answer_extraction_tag: str = "####",
    answer_extraction_instruction: str = "Output your final answer after ####",
    # Optional fields
    majority_voting_method: Optional[str] = None,
    answer_comparison_method: Optional[str] = None,
    translator_model_path: Optional[str] = None,
) -> dict:
    """Create configuration dictionary from UI inputs"""

    # Convert string inputs to appropriate types
    try:
        num_samples = int(num_samples) if num_samples else 10
    except (ValueError, TypeError):
        num_samples = 10

    # Base configuration structure
    config = {
        "seed": 2024,
        "device": cuda_device,
        "output_dir": output_dir or "./outputs",
        "export_format": "jsonl",
        "task": {
            "task_type": task_type.lower() if task_type else "local",  # Convert to lowercase: Local -> local, Web -> web, Cloud -> cloud
            "name": task_name or "custom_task",
            "domain": domain or "general",
        },
        "llm": {
            "provider": llm_provider,
            "model": llm_model,
            "api_key": llm_api_key if llm_api_key else None,
            "base_url": llm_base_url if llm_base_url else None,
        },
        "base_model": {
            "provider": "local",
            "path": base_model_path,
            # Device is injected globally via config.device by from_yaml()
            "inference": {
                "temperature": 0.0,
                "max_tokens": 1500,
                "top_p": 0.95,
                "n": 1
            },
            "scoring": {
                "temperature": 1.2,
                "max_tokens": 1500,
                "top_p": 0.95,
                "n": 12
            }
        },
        "translation": {
            "language": language,
            "model_path": translator_model_path if translator_model_path and language.lower() == "arabic" else None,
            "max_tokens": 256,
            "batch_size": 1
        },
        "answer_extraction": {
            "tag": answer_extraction_tag,
            "instruction": answer_extraction_instruction
        },
        "postprocess": {
            "methods": ["majority_voting"],
            "majority_voting": {
                "n": 4,
                "method": majority_voting_method or "exact_match",
                "exact_match": {
                    "numeric_tolerance": 1e-3
                }
            }
        },
        "evaluation": {
            "batch_size": 100,
            "answer_comparison": {
                "method": answer_comparison_method or "semantic",
            }
        },
        "rewrite": {
            "method": "difficulty_adjust",
            "input_instruction": input_instruction,
            "output_instruction": output_instruction,
            "difficulty_adjust": {
                "easier_temperature": 0.9,
                "harder_temperature": 1.1
            }
        }
    }

    # Configure majority voting based on method
    if majority_voting_method == "semantic_clustering":
        config["postprocess"]["majority_voting"]["semantic_clustering"] = {
            "model_path": semantic_model_path,
            "similarity_threshold": 0.85
            # Device will use default "cuda:0" from SemanticClusteringVotingConfig
        }
    # Other methods (exact_match, llm_judge) will use their default values

    # Configure answer comparison based on method
    if answer_comparison_method == "semantic":
        config["evaluation"]["answer_comparison"]["semantic"] = {
            "model_path": semantic_model_path,
            "similarity_threshold": 0.85
            # Device will use default "cuda:0" from SemanticComparisonConfig
        }
    elif answer_comparison_method == "exact_match":
        config["evaluation"]["answer_comparison"]["exact_match"] = {
            "numeric_tolerance": 1e-3
        }

    # Task-specific configuration
    task_type_lower = task_type.lower() if task_type else "local"

    if task_type_lower == "local":
        # Create temp directory for uploaded documents
        temp_doc_dir = os.path.join(tempfile.gettempdir(), f"sdg_docs_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(temp_doc_dir, exist_ok=True)

        # Copy uploaded files
        if documents:
            for doc in documents:
                if doc is not None:
                    # Convert Gradio file object to string path
                    doc_path = str(doc) if hasattr(doc, '__str__') else doc
                    shutil.copy(doc_path, temp_doc_dir)

        config["task"]["local"] = {
            "parsing": {
                "document_dir": temp_doc_dir,
                "method": "mineru",  # Default parser method
                # Device is injected globally via config.device by from_yaml()
            },
            "retrieval": {
                "passages_dir": os.path.join(output_dir or "./outputs", "passages"),
                "method": "bm25",
                "top_k": 10000
            },
            "generation": {
                "task_instruction": task_instruction,
                "input_instruction": input_instruction,
                "output_instruction": output_instruction,
                "num_samples": num_samples,
                "temperature": 1.0
            }
        }

    elif task_type_lower == "web":
        config["task"]["web"] = {
            "task_instruction": task_instruction,
            "input_instruction": input_instruction,
            "output_instruction": output_instruction,
            "dataset_limit": 1,
            "num_samples": num_samples,
            "dataset_score_threshold": dataset_score_threshold
        }
        if hf_token:
            config["task"]["web"]["huggingface_token"] = hf_token

    elif task_type_lower == "distill":
        config["task"]["distill"] = {
            "task_instruction": task_instruction,
            "input_instruction": input_instruction,
            "output_instruction": output_instruction,
            "num_samples": num_samples,
            "batch_size": 5,
            "temperature": 1.0
        }

    return config


def generate_data(
    # Essential fields
    task_type: str,
    task_instruction: str,
    input_instruction: str,
    output_instruction: str,
    num_samples: int,
    llm_provider: str,
    llm_model: str,
    llm_api_key: str,
    llm_base_url: str,
    # Basic settings
    task_name: str,
    domain: str,
    output_dir: str,
    cuda_device: str,
    hf_token: str,
    language: str,
    # Model paths
    base_model_path: str,
    semantic_model_path: str,
    # Local task specific
    documents: Optional[List] = None,
    # Web task specific
    dataset_score_threshold: int = 30,
    # Answer extraction
    answer_extraction_tag: str = "####",
    answer_extraction_instruction: str = "Output your final answer after ####",
    # Optional fields
    majority_voting_method: str = "exact_match",
    answer_comparison_method: str = "semantic",
    translator_model_path: str = ""
):
    """
    Main function to generate synthetic data with real-time step progress tracking

    Yields:
        Tuple of (status_banner, progress_cards, output_files, output_section_visible) at each update
    """
    import threading
    import time

    # Clear previous logs
    log_capture.clear()

    # Initialize step tracking
    current_step_index = [-1]  # -1 means not started

    try:
        logger.info("=" * 80)
        logger.info("Starting DataArc Synthetic Data Generation")
        logger.info("=" * 80)

        # Validate inputs
        if llm_provider in ["openai", "ollama"] and not llm_api_key:
            error_msg = f"API Key is required for {llm_provider} provider!"
            logger.error(error_msg)
            # Show error banner and message
            yield (
                create_error_banner_html(error_msg),
                create_error_message_html(error_msg),
                "",
                gr.update(visible=True),
                None
            )
            return

        if task_type and task_type.lower() == "local" and not documents:
            error_msg = "Please upload documents for local task type!"
            logger.error(error_msg)
            yield (
                create_error_banner_html(error_msg),
                create_error_message_html(error_msg),
                "",
                gr.update(visible=True),
                None
            )
            return

        logger.info(f"Task Type: {task_type}")
        logger.info(f"Task Name: {task_name}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"LLM Provider: {llm_provider}")
        logger.info(f"LLM Model: {llm_model}")
        logger.info(f"Device: {cuda_device}")
        logger.info(f"Base Model: {base_model_path}")
        logger.info(f"Answer Comparison: {answer_comparison_method}")
        logger.info(f"Language: {language}")

        # Yield initial state - waiting for steps to be detected from logs
        current_step_index[0] = -1
        yield (
            create_status_banner_html("in-progress", num_samples, llm_model, False),
            create_progress_cards_html([], -1),  # Start with empty steps, will populate from logs
            "",
            gr.update(visible=True),
            None
        )

        # Create configuration
        config_dict = create_config_dict(
            task_type=task_type,
            task_instruction=task_instruction,
            input_instruction=input_instruction,
            output_instruction=output_instruction,
            num_samples=num_samples,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_api_key=llm_api_key,
            llm_base_url=llm_base_url,
            task_name=task_name,
            domain=domain,
            output_dir=output_dir,
            cuda_device=cuda_device,
            hf_token=hf_token,
            language=language,
            base_model_path=base_model_path,
            semantic_model_path=semantic_model_path,
            documents=documents,
            dataset_score_threshold=dataset_score_threshold,
            answer_extraction_tag=answer_extraction_tag,
            answer_extraction_instruction=answer_extraction_instruction,
            majority_voting_method=majority_voting_method,
            answer_comparison_method=answer_comparison_method,
            translator_model_path=translator_model_path
        )

        # Save config to temp file for debugging
        config_path = os.path.join(tempfile.gettempdir(), f"sdg_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml")
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"Configuration saved to: {config_path}")

        # Parse configuration
        temp_config_file = os.path.join(tempfile.gettempdir(), "temp_config.yaml")
        with open(temp_config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, allow_unicode=True)

        config = SDGSConfig.from_yaml(temp_config_file)
        logger.info("Configuration parsed successfully")

        # Initialize pipeline
        pipeline = Pipeline(config)
        logger.info("Pipeline initialized successfully")

        # Create output directory
        os.makedirs(output_dir or "./outputs", exist_ok=True)

        # Run pipeline in background thread with dynamic log-based progress tracking
        logger.info("Starting data generation pipeline...")

        pipeline_error: List[Optional[Exception]] = [None]
        pipeline_done = [False]

        def run_pipeline():
            try:
                # Run pipeline without callback - we'll parse logs instead
                pipeline.run()
            except Exception as e:
                pipeline_error[0] = e
            finally:
                pipeline_done[0] = True

        pipeline_thread = threading.Thread(target=run_pipeline, daemon=True)
        pipeline_thread.start()

        # Dynamic step tracking from logs
        import re
        detected_steps = []  # List of detected step descriptions
        current_step_index[0] = -1  # -1 means not started
        step_pattern = re.compile(r"=== Step: (.+?) ===")
        usage_pattern = re.compile(r"\[(.+?) Usage\]: completed=(\d+), token=(\d+), time=([\d.]+) \|\| remain=(\d+), remain_token_anticipation=(\d+), remain_time_anticipatioin=([\d.]+)")
        model_loading_pattern = re.compile(r"Loading .+? model", re.IGNORECASE)
        parsing_pattern = re.compile(r"Parsing document with .+? model", re.IGNORECASE)

        # Web task dataset patterns
        searched_dataset_pattern = re.compile(r"  - (.+?) \(keyword:")
        final_dataset_pattern = re.compile(r"  Dataset (.+?): will extract (\d+) samples")

        # Separate tracking for different message types
        current_model_loading = [None]  # Model loading status
        current_parsing = [None]  # Document parsing status
        current_usage_counter = [None]  # Usage counter progress

        # Track completed step statistics and step-specific info
        completed_step_stats = {}  # {step_index: "stats string"}
        step_dataset_info = {}  # {step_index: "dataset list string"} - for searched/final datasets
        step_usage_tracking = {}  # {step_name: {'step_idx': int, 'token': int, 'time': float}}
        last_usage_step_name = [None]  # Track which step name was last processing

        # Stream progress updates while pipeline is running
        while not pipeline_done[0]:
            time.sleep(0.5)

            # Get new logs since last check
            new_logs = log_capture.get_new_logs()

            if new_logs:
                # Check for parsing messages
                parsing_match = parsing_pattern.search(new_logs)
                if parsing_match:
                    parsing_msg = parsing_match.group(0)
                    new_parsing_msg = f"{parsing_msg}..."
                    # Only update and log if it's a new message (avoid infinite loop)
                    if current_parsing[0] != new_parsing_msg:
                        current_parsing[0] = new_parsing_msg
                        # Clear loading message when parsing starts
                        current_model_loading[0] = None
                        logger.info(f"[UI Progress] Set parsing message: {parsing_msg}")

                # Check for model loading messages
                model_loading_match = model_loading_pattern.search(new_logs)
                if model_loading_match:
                    loading_msg = model_loading_match.group(0)
                    new_loading_msg = f"{loading_msg}..."
                    # Only update and log if it's a new message (avoid infinite loop)
                    if current_model_loading[0] != new_loading_msg:
                        current_model_loading[0] = new_loading_msg
                        logger.info(f"[UI Progress] Detected model loading: {loading_msg}")

                # Check for web task searched datasets (collect dataset IDs only)
                # Store under current step index
                searched_matches = list(searched_dataset_pattern.finditer(new_logs))
                if searched_matches:
                    datasets_info = [match.group(1) for match in searched_matches]
                    if datasets_info:
                        dataset_list_text = "Searched Datasets:\n" + "\n".join(f"  • {ds}" for ds in datasets_info)
                        # Store for current step
                        step_dataset_info[current_step_index[0]] = dataset_list_text

                # Check for web task final datasets to use (collect all dataset entries with sample counts)
                # Store under current step index
                dataset_matches = list(final_dataset_pattern.finditer(new_logs))
                if dataset_matches:
                    datasets_info = []
                    for match in dataset_matches:
                        dataset_id = match.group(1)
                        samples = match.group(2)
                        datasets_info.append(f"{dataset_id} ({samples} samples)")

                    if datasets_info:
                        dataset_list_text = "Final Datasets to Use:\n" + "\n".join(f"  • {ds}" for ds in datasets_info)
                        # Store for current step
                        step_dataset_info[current_step_index[0]] = dataset_list_text

                # Parse for usage counter information (before step markers)
                usage_matches = list(usage_pattern.finditer(new_logs))
                if usage_matches:
                    # Get the most recent usage info
                    latest_match = usage_matches[-1]
                    step_name = latest_match.group(1)  # e.g., "LocalTask-Generation"
                    completed = int(latest_match.group(2))
                    token = int(latest_match.group(3))
                    time_spent = float(latest_match.group(4))
                    remain = int(latest_match.group(5))
                    remain_token = int(latest_match.group(6))
                    remain_time = float(latest_match.group(7))

                    # Track usage for current step
                    if step_name not in step_usage_tracking:
                        step_usage_tracking[step_name] = {
                            'step_idx': current_step_index[0],
                            'token': token,
                            'time': time_spent
                        }
                    else:
                        # Update accumulated values
                        step_usage_tracking[step_name]['token'] = token
                        step_usage_tracking[step_name]['time'] = time_spent

                    last_usage_step_name[0] = step_name

                    # Format usage info for display (real-time progress)
                    total = completed + remain
                    current_usage_counter[0] = f"[{step_name}]\nProgress: {completed}/{total} | Tokens: {token} | Time: {time_spent:.2f}s\nRemaining: ~{remain} iterations | ~{remain_token} tokens | ~{remain_time:.2f}s"

                    # Clear loading/parsing messages when usage counter starts
                    # Keep dataset list visible
                    current_model_loading[0] = None
                    current_parsing[0] = None

                # Parse for step markers (after usage)
                for match in step_pattern.finditer(new_logs):
                    step_description = match.group(1)
                    # Add new step if not already detected
                    if step_description not in detected_steps:
                        # Before moving to new step, finalize stats for step that just completed
                        # The last usage we saw belongs to the step that just finished
                        if last_usage_step_name[0] and last_usage_step_name[0] in step_usage_tracking:
                            data = step_usage_tracking[last_usage_step_name[0]]
                            # Store stats for the step that the usage belongs to
                            completed_step_stats[data['step_idx']] = f"[{last_usage_step_name[0]}] Total\nTokens: {data['token']} | Time: {data['time']:.2f}s"

                        detected_steps.append(step_description)
                        current_step_index[0] = len(detected_steps) - 1

                        # Clear all messages when switching to new step
                        # Keep dataset list visible throughout the pipeline
                        current_usage_counter[0] = None
                        current_parsing[0] = None
                        current_model_loading[0] = None

                        logger.info(f"[UI Progress] Detected step {len(detected_steps)}: {step_description}")

            # Combine messages with priority: usage counter > parsing > model loading
            display_message = current_usage_counter[0] or current_parsing[0] or current_model_loading[0]

            # Yield progress update with dynamic steps, message, and dataset info
            yield (
                create_status_banner_html("in-progress", num_samples, llm_model, False),
                create_progress_cards_html(detected_steps, current_step_index[0], display_message, completed_step_stats, step_dataset_info),
                "",
                gr.update(visible=True),
                None
            )

        pipeline_thread.join(timeout=1)

        if pipeline_error[0]:
            raise pipeline_error[0]

        logger.info("=" * 80)
        logger.info("Data generation completed successfully!")
        logger.info("=" * 80)

        # Generate output files HTML
        output_files_html = create_output_files_html(
            output_dir or "./outputs",
            task_name,
            config.export_format
        )

        # Determine which file to provide for download (priority: learnable > solved > unsolved)
        base_path = f"{output_dir or './outputs'}/{task_name}_final"
        download_file_path = None

        learnable_file = f"{base_path}_learnable.{config.export_format}"
        solved_file = f"{base_path}_solved.{config.export_format}"
        unsolved_file = f"{base_path}_unsolved.{config.export_format}"

        if os.path.exists(learnable_file):
            download_file_path = os.path.abspath(learnable_file)
        elif os.path.exists(solved_file):
            download_file_path = os.path.abspath(solved_file)
        elif os.path.exists(unsolved_file):
            download_file_path = os.path.abspath(unsolved_file)

        # Final yield - all steps completed with download file
        # Mark all detected steps as completed and finalize last step stats
        last_step_idx = len(detected_steps) - 1
        if last_step_idx >= 0 and step_usage_tracking:
            for step_name, data in step_usage_tracking.items():
                if data['step_idx'] == last_step_idx:
                    completed_step_stats[last_step_idx] = f"[{step_name}] Total\nTokens: {data['token']} | Time: {data['time']:.2f}s"
                    break

        yield (
            create_status_banner_html("completed", num_samples, llm_model, True, download_file_path),
            create_progress_cards_html(detected_steps, len(detected_steps), None, completed_step_stats, step_dataset_info),  # All detected steps completed with stats and dataset info
            output_files_html,
            gr.update(visible=True),
            download_file_path  # Provide file to gr.File component
        )

    except Exception as e:
        error_msg = f"Error during generation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        # Show error banner and detailed error message
        yield (
            create_error_banner_html(error_msg),
            create_error_message_html(error_msg),
            "",
            gr.update(visible=True),
            None
        )


def update_steps_based_on_task_type(_task_type: str):  # noqa: ARG001
    """Keep wizard step count consistent (always 3 steps)."""
    # This function is used only for triggering JS callback
    # No output needed since outputs=[]
    return None


def update_semantic_visibility(majority_voting_method: str, answer_comparison_method: str):
    """Update semantic model path visibility based on both majority voting and answer comparison methods"""
    # Show semantic model path if either method uses semantic
    if answer_comparison_method == "semantic" or majority_voting_method == "semantic_clustering":
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def update_hf_token_visibility(task_type: str):
    """Show HF token and parser options based on task type"""
    normalized = (task_type or "Local").lower()
    is_web = normalized == "web"
    is_local = normalized == "local"
    return (
        gr.update(visible=is_local),  # documents_group
        gr.update(visible=is_local),  # parser_method_group
        gr.update(visible=is_web),    # hf_token_group
        gr.update(visible=is_web),    # dataset_score_threshold_group
        gr.update(visible=not is_web) # demo_samples_group (hide for web task)
    )


def update_translator_visibility(language: str):
    """Show translator model path when Arabic is selected"""
    normalized = (language or "English").lower()
    is_arabic = normalized == "arabic"
    return gr.update(visible=is_arabic)


with gr.Blocks(title="DataArc SDG - Synthetic Data Generator", theme=gr.themes.Soft(), css=CUSTOM_CSS, head=CUSTOM_HEAD, elem_classes=["carbon-body"], fill_width=True) as demo:
    gr.HTML(
        f"""
        <header class="carbon-header">
            <div class="carbon-header__top">
                <div class="carbon-header__layout">
                    <div class="carbon-header__left">
                        <div class="brand-container">
                            <img src="{LOGO_BASE64}" alt="DataArc Logo" class="brand-logo" />
                            <span class="brand-text">DataArc Synthetic Data Generator</span>
                        </div>
                    </div>
                    <div class="carbon-header__nav">
                        <div class="header-nav">
                            <button class="header-nav__item active" data-nav="configuration" type="button">
                                <i class="iconfont icon-synthetic-a-NavConfigurationIcon"></i>
                                Configuration
                            </button>
                            <button class="header-nav__item disabled" data-nav="generate" type="button">
                                <i class="iconfont icon-synthetic-a-Group11"></i>
                                Generate Dataset
                            </button>
                        </div>
                    </div>
                    <div class="carbon-header__right">
                        <button id="header_config_btn" class="carbon-btn config drawer-trigger" type="button">
                            <span class="iconfont icon-synthetic-Setting"></span>
                            LLM Settings
                        </button>
                    </div>
                </div>
            </div>
            <div class="carbon-header__bottom" aria-hidden="true">
                <div class="carbon-header__layout" style="justify-content: center;">
                    <div class="header-nav">
                        <button class="header-nav__item active" type="button">
                            <span class="nav-dot">
                                <i class="iconfont icon-synthetic-a-GenerationStepDefault"></i>
                                <span class="nav-dot-number">1</span>
                            </span>
                            Basic Settings
                        </button>
                        <button class="header-nav__item" type="button">
                            <span class="nav-dot">
                                <i class="iconfont icon-synthetic-a-GenerationStepDefault"></i>
                                <span class="nav-dot-number">2</span>
                            </span>
                            Task Configuration
                        </button>
                        <button class="header-nav__item" type="button">
                            <span class="nav-dot">
                                <i class="iconfont icon-synthetic-a-GenerationStepDefault"></i>
                                <span class="nav-dot-number">3</span>
                            </span>
                            Advanced Settings
                        </button>
                    </div>
                </div>
            </div>
        </header>
        <div id="drawer_portal">
            <button id="settings_toggle" class="iconfont icon-shezhi" title="打开高级设置" style="display:none;"></button>
            <div id="drawer_overlay"></div>
        </div>
        """
    )

    with gr.Column(elem_id="main_layout"):
        with gr.Column(elem_id="config_section"):
            with gr.Group(elem_id="left_column"):

                # Step 1: Basic Settings
                with gr.Group(elem_classes=["form-section", "step-content", "active"], elem_id="step_1"):
                    # Task Name
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Task Name</span>
                            </div>
                            <div class="field-label-info">Name this synthetic data generation task.</div>
                        """)
                        task_name = gr.Textbox(
                            label="",
                            placeholder="custom_task",
                            value="custom_task",
                            show_label=False
                        )

                    # Domain
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Domain</span>
                            </div>
                            <div class="field-label-info">Specify the domains of the dataset to help generate more relevant content.</div>
                        """)
                        domain = gr.Textbox(
                            label="",
                            placeholder="general",
                            value="general",
                            show_label=False
                        )

                    # Output Directory
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Output Directory</span>
                            </div>
                            <div class="field-label-info">Save path for generated data, the system will create dataset files here.</div>
                        """)
                        output_dir = gr.Textbox(
                            label="",
                            placeholder="./output",
                            value="./output",
                            show_label=False
                        )

                    # CUDA Device (Optional)
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon">
                                <span class="field-label-text">CUDA Device</span>
                            </div>
                            <div class="field-label-info">Specify the CUDA device used for parsing, scoring, and evaluation.</div>
                        """)
                        cuda_device = gr.Textbox(
                            label="",
                            placeholder="e.g., 0 or cuda:0",
                            value="cuda:0",
                            show_label=False,
                            elem_classes=["optional-field"]
                        )

                # Step 2: Task Configuration
                with gr.Group(elem_classes=["form-section", "step-content"], elem_id="step_2"):
                    # Custom Task Type with cards
                    gr.HTML("""
                        <div style="display: flex; flex-direction: column; gap: 3.5px; margin-bottom: 20px; background: #FFFFFF; padding: 4px 0;">
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Task Type</span>
                            </div>
                            <span style="font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; font-size: 10.5px; line-height: 1.34; letter-spacing: 0.0305em; color: #525252;">
                                Choose synthetic data generation method.
                            </span>
                        </div>
                    """)

                    with gr.Group(elem_id="task_type_cards", elem_classes=["task-type-container"]):
                        gr.HTML("""
                            <div class="task-type-options">
                                <div class="task-type-card active" data-value="Local" onclick="window.__selectTaskType && window.__selectTaskType('Local', this)">
                                    <span class="iconfont icon-synthetic-Local task-type-icon"></span>
                                    <div class="task-type-content">
                                        <div class="task-type-title">Local</div>
                                        <div class="task-type-desc">Generate synthetic data based on uploaded documents.</div>
                                    </div>
                                    <div class="task-type-check"></div>
                                </div>
                                <div class="task-type-card" data-value="Web" onclick="window.__selectTaskType && window.__selectTaskType('Web', this)">
                                    <span class="iconfont icon-synthetic-Web task-type-icon"></span>
                                    <div class="task-type-content">
                                        <div class="task-type-title">Web</div>
                                        <div class="task-type-desc">Generate synthetic data based on opensource dataset.</div>
                                    </div>
                                    <div class="task-type-check"></div>
                                </div>
                                <div class="task-type-card" data-value="Distill" onclick="window.__selectTaskType && window.__selectTaskType('Distill', this)">
                                    <span class="iconfont icon-synthetic-Distill task-type-icon"></span>
                                    <div class="task-type-content">
                                        <div class="task-type-title">Distill</div>
                                        <div class="task-type-desc">Generate synthetic data through knowledge distillation from large language models.</div>
                                    </div>
                                    <div class="task-type-check"></div>
                                </div>
                            </div>
                        """)
                        task_type = gr.Radio(
                            choices=["Local", "Web", "Distill"],
                            value="Local",
                            show_label=False,
                            elem_id="task_type_hidden",
                            elem_classes=["task-type-hidden"]
                        )

                    # Task Instruction
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Task Instruction</span>
                            </div>
                            <div class="field-label-info">Clearly describe the data you want to generate. This will guide the entire generation process.</div>
                        """)
                        task_instruction = gr.Textbox(
                            label="",
                            placeholder="Describe the data you want to generate",
                            lines=3,
                            value="Generate a grade school math word problem that requires multi-step reasoning. The problem should involve basic arithmetic operations and have a clear numerical answer.",
                            show_label=False
                        )

                    # Input Instruction
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon">
                                <span class="field-label-text">Input Instruction</span>
                            </div>
                            <div class="field-label-info">Specify the input format of the data you want to generate. This will guide the entire generation process.</div>
                        """)
                        input_instruction = gr.Textbox(
                            label="",
                            placeholder="Specify the input format",
                            lines=2,
                            value="The question should be a math word problem that requires step by step reasoning.",
                            show_label=False,
                            elem_classes=["optional-field"]
                        )

                    # Output Instruction
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon">
                                <span class="field-label-text">Output Instruction</span>
                            </div>
                            <div class="field-label-info">Specify the expected output format of the data you want to generate. This will guide the entire generation process.</div>
                        """)
                        output_instruction = gr.Textbox(
                            label="",
                            placeholder="Specify the output format",
                            lines=2,
                            value="The output should contain reasoning process step by step.",
                            show_label=False,
                            elem_classes=["optional-field"]
                        )

                    # Answer Extraction Configuration
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon">
                                <span class="field-label-text">Answer Extraction</span>
                            </div>
                            <div class="field-label-info">Configure how the final answer is extracted from the output of the data for evaluation.</div>
                        """)

                        answer_extraction_tag = gr.Textbox(
                            label="Extraction Tag",
                            placeholder="e.g., ####, <answer>, [ANSWER]",
                            value="####",
                            show_label=False,
                            elem_classes=["optional-field"],
                            info="Tag used to mark the final answer in output"
                        )

                        answer_extraction_instruction = gr.Textbox(
                            label="Extraction Instruction",
                            placeholder="Instructions for answer extraction",
                            value="Output your final answer after ####",
                            show_label=False,
                            elem_classes=["optional-field"],
                            info="Instructions on the format of final answer in output"
                        )

                    # Number of samples
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Number of samples</span>
                            </div>
                            <div class="field-label-info">Total number of data samples to generate. It's recommended to start with a small scale for testing.</div>
                        """)
                        num_samples = gr.Textbox(
                            label="",
                            placeholder="10",
                            value="10",
                            show_label=False
                        )

                    # Custom file upload with Figma design (Local-only)
                    with gr.Group(elem_classes=["file-upload-container"], elem_id="documents_group", visible=True) as documents_group:
                        gr.HTML("""
                            <div style="display: flex; flex-direction: column; gap: 3.5px; margin-bottom: 20px; width: 100%; text-align: left; background: #FFFFFF;">
                                <div class="field-label-with-icon required">
                                    <span class="field-label-text">Upload Documents</span>
                                </div>
                                <span style="font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; font-size: 12px; line-height: 1.17; letter-spacing: 0.32px; color: #525252; width: 100%; display: block; overflow: visible; white-space: normal; word-wrap: break-word;">
                                    Upload local documents for data generation.
                                </span>
                            </div>
                        """)
                        documents = gr.File(
                            label="",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md"],
                            type="filepath",
                            show_label=False
                        )

                    with gr.Group(elem_classes=["form-field"], elem_id="parser_method_group", visible=True) as parser_method_group:
                        gr.HTML("""
                            <div class="field-label-with-icon">
                                <span class="field-label-text">Parser Method</span>
                            </div>
                            <div class="field-label-info">Select parser to parse local documents.</div>
                        """)
                        parser_method = gr.Dropdown(
                            label="",
                            choices=["mineru"],
                            value="mineru",
                            show_label=False,
                            interactive=True
                        )

                    with gr.Group(elem_classes=["form-field"], elem_id="hf_token_group", visible=False) as hf_token_group:
                        hf_token_label = gr.HTML(
                            """
                            <div class="field-label-with-icon">
                                <span class="field-label-text">HuggingFace Token (Optional)</span>
                            </div>
                            <div class="field-label-info">Specify the HuggingFace token for accessing private models</div>
                            """
                        )
                        hf_token = gr.Textbox(
                            label="",
                            placeholder="Enter your HuggingFace token",
                            value="",
                            type="password",
                            show_label=False,
                            elem_classes=["optional-field"]
                        )

                    with gr.Group(elem_classes=["form-field"], elem_id="dataset_score_threshold_group", visible=False) as dataset_score_threshold_group:
                        gr.HTML(
                            """
                            <div class="field-label-with-icon">
                                <span class="field-label-text">Dataset Score Threshold</span>
                            </div>
                            <div class="field-label-info">Minimum overall score (sum of 5 criteria) for a dataset to be valid (0-50)</div>
                            """
                        )
                        dataset_score_threshold = gr.Number(
                            label="",
                            value=30,
                            minimum=0,
                            maximum=50,
                            step=1,
                            show_label=False
                        )


                # Step 3: Advanced Settings
                with gr.Group(elem_classes=["form-section", "step-content"], elem_id="step_3"):

                    with gr.Group(elem_classes=["file-upload-container"], elem_id="demo_samples_group") as demo_samples_group:
                        gr.HTML("""
                            <div style="display: flex; flex-direction: column; gap: 3.5px; margin-bottom: 20px; width: 100%; text-align: left; background: #FFFFFF;">
                                <div style="display: flex; align-items: center; gap: 4px;">
                                    <span style="font-family: 'IBM Plex Sans', sans-serif; font-weight: 500; font-size: 14px; line-height: 1.3; color: #161616; display: block;">
                                        Demo Example (Optional)
                                    </span>
                                </div>
                                <span style="font-family: 'IBM Plex Sans', sans-serif; font-weight: 400; font-size: 12px; line-height: 1.17; letter-spacing: 0.32px; color: #525252; width: 100%; display: block; overflow: visible; white-space: normal; word-wrap: break-word;">
                                    Upload the expected data demo to guide the data generation.
                                </span>
                            </div>
                        """)
                        demo_samples = gr.File(
                            label="",
                            file_count="multiple",
                            file_types=[".jsonl"],
                            type="filepath",
                            show_label=False
                        )

                    # Majority Voting Method
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Majority Voting Method</span>
                            </div>
                            <div class="field-label-info">Method to determine the ground truth when synthesize data using LLM.</div>
                        """)
                        majority_voting_method = gr.Dropdown(
                            label="",
                            choices=["exact_match", "semantic_clustering", "llm_judge"],
                            value="exact_match",
                            show_label=False
                        )

                    # Answer Comparison Method
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Answer Comparison Method</span>
                            </div>
                            <div class="field-label-info">Method to compare the answer between base model answer and ground truth for evaluation.</div>
                        """)
                        answer_comparison_method = gr.Dropdown(
                            label="",
                            choices=["exact_match", "semantic", "llm_judge"],
                            value="exact_match",
                            show_label=False
                        )

                    # Semantic Model Path (hidden by default, shown when semantic or semantic_clustering is selected)
                    with gr.Group(elem_classes=["form-field"], visible=False) as semantic_model_path_group:
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Semantic Model Path</span>
                            </div>
                            <div class="field-label-info">Specify the path to the semantic model for semantic similarity comparison.</div>
                        """)
                        semantic_model_path = gr.Textbox(
                            label="",
                            placeholder="BAAI/bge-small-en-v1.5",
                            value="BAAI/bge-small-en-v1.5",
                            show_label=False
                        )

                    # Dataset Language moved here
                    with gr.Group(elem_classes=["form-field"]):
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Dataset Language</span>
                            </div>
                            <div class="field-label-info">Select the language of the output dataset.</div>
                        """)
                        language = gr.Dropdown(
                            label="",
                            choices=["English", "Arabic"],
                            value="English",
                            show_label=False
                        )

                    # Arabic Translator Model Path (conditional - only visible when Arabic is selected)
                    with gr.Group(elem_classes=["form-field"], elem_id="translator_model_group", visible=False) as translator_model_group:
                        gr.HTML("""
                            <div class="field-label-with-icon required">
                                <span class="field-label-text">Arabic Translator Model Path</span>
                            </div>
                            <div class="field-label-info">Specify the path to the translator model for translation.</div>
                        """)
                        translator_model_path = gr.Textbox(
                            label="",
                            placeholder="Enter the path to Arabic translator model",
                            value="",
                            show_label=False
                        )

                # Navigation Buttons
                with gr.Row(elem_classes=["step-navigation"]):
                    btn_previous = gr.Button("← Previous", elem_id="btn_previous", size="lg", visible=True)
                    btn_next = gr.Button("Next →", elem_id="btn_next", size="lg", variant="primary", visible=True)
                    generate_btn = gr.Button("Generate Dataset", elem_id="btn_submit", size="lg", variant="primary", visible=True)

        # Output Section (initially hidden)
        output_section = gr.Column(elem_id="output_section", visible=False)
        with output_section:
            # Generation Status Banner
            generation_status_banner = gr.HTML(
                value="",
                elem_id="generation_status_banner"
            )

            # Progress Cards Container
            progress_cards_container = gr.HTML(
                value="",
                elem_id="progress_cards_html"
            )

            # Output Files Section (initially hidden via HTML)
            output_files_display = gr.HTML(
                value="",
                elem_id="output_files_display"
            )

            # Hidden download file component (for programmatic download via button)
            # visible=True keeps it in DOM, CSS hides it visually
            download_file = gr.File(
                label="Download Generated Data",
                visible=True,
                interactive=False,
                elem_id="hidden_download_file"
            )

    with gr.Column(elem_id="settings_drawer"):
        gr.HTML(
            """
            <div class="drawer-header">
                <div style="display: flex; flex-direction: column; gap: 6px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="padding: 8px; border-radius: 2px; background: rgba(15, 98, 254, 0.1); display: inline-flex; width: 32px; height: 32px; align-items: center; justify-content: center;">
                            <i class="iconfont icon-synthetic-Setting-active" style="font-size: 20px;"></i>
                        </div>
                        <h2 style="margin: 0; font-size: 18px; font-weight: 500; color: #161616; letter-spacing: 0;">LLM Settings</h2>
                    </div>
                    <p style="margin: 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; font-weight: 400; line-height: 1.33; color: #525252;">Manage LLM model configuration and basic parameters.</p>
                </div>
                <button id="drawer_close" aria-label="close settings">×</button>
            </div>
            """
        )

        with gr.Column(elem_classes=["drawer-body"]):
            # LLM Configuration Section
            gr.HTML(
                """
                <div style="display: flex; flex-direction: column; gap: 4px; padding: 16px; border-left: 2px solid #524AC9; margin-bottom: 16px;">
                    <h3 style="margin: 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; font-weight: 500; line-height: 1.3; color: #161616;">LLM Configuration</h3>
                    <p style="margin: 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; font-weight: 400; line-height: 1.3; color: #525252;">Configure the language model for data generation</p>
                </div>
                """
            )

            llm_provider = gr.Dropdown(
                label="LLM Provider",
                choices=["openai", "ollama", "anthropic"],
                value="openai"
            )
            llm_model = gr.Textbox(
                label="LLM Model",
                placeholder="model name",
                value="gpt-4o-mini",
                info="e.g., gpt-4o-mini, claude-3-opus, gemini-pro."
            )
            llm_api_key = gr.Textbox(
                label="API Key",
                placeholder="API Key String",
                type="password",
                info="Your API KEY to access the model"
            )
            llm_base_url = gr.Textbox(
                label="Base URL (Optional)",
                placeholder="https://api.openai.com/v1",
                value="",
                info="Custom API endpoint address.",
                elem_classes=["optional-field"]
            )

            # Base Model Configuration Section
            gr.HTML(
                """
                <div style="display: flex; flex-direction: column; gap: 4px; padding: 16px; border-left: 2px solid #524AC9; margin-top: 24px; margin-bottom: 16px;">
                    <h3 style="margin: 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 14px; font-weight: 500; line-height: 1.3; color: #161616;">Base Model Configuration</h3>
                    <p style="margin: 0; font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; font-weight: 400; line-height: 1.3; color: #525252;">Local model path for evaluation</p>
                </div>
                """
            )

            base_model_path = gr.Textbox(
                label="Base Model Path",
                value="Qwen/Qwen2.5-7B",
                placeholder="Qwen/Qwen2.5-7B",
                info="Specify path for local model file."
            )

        # Drawer Footer
        gr.HTML(
            """
            <div class="drawer-footer">
                <button onclick="document.getElementById('drawer_close').click()" style="width: 100%; border-radius: 0; border: none; background: #524AC9; color: white; height: 48px; font-weight: 500; cursor: pointer; font-size: 16px; transition: background 0.11s cubic-bezier(0.2, 0, 0.38, 0.9);" onmouseover="this.style.background='#0353e9'" onmouseout="this.style.background='#524AC9'">
                    Close
                </button>
            </div>
            """
        )

    # Step navigation button handlers
    btn_next.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { window.__sdgNextStep(); return []; }"
    )

    btn_previous.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { window.__sdgPreviousStep(); return []; }"
    )

    # Update step count based on task type
    task_type.change(
        fn=update_steps_based_on_task_type,
        inputs=[task_type],
        outputs=[],
        js="""
        (task_type) => {
            if (window.__sdgSetTotalSteps) {
                window.__sdgSetTotalSteps(3);
            }
            return [];
        }
        """
    )

    task_type.change(
        fn=update_hf_token_visibility,
        inputs=[task_type],
        outputs=[documents_group, parser_method_group, hf_token_group, dataset_score_threshold_group, demo_samples_group]
    )

    # Update semantic fields visibility when answer comparison method changes
    answer_comparison_method.change(
        fn=update_semantic_visibility,
        inputs=[majority_voting_method, answer_comparison_method],
        outputs=[semantic_model_path_group]
    )

    # Update semantic fields visibility when majority voting method changes
    majority_voting_method.change(
        fn=update_semantic_visibility,
        inputs=[majority_voting_method, answer_comparison_method],
        outputs=[semantic_model_path_group]
    )

    # Update translator model path visibility based on language
    language.change(
        fn=update_translator_visibility,
        inputs=[language],
        outputs=[translator_model_group]
    )

    # Generate button click
    generate_btn.click(
        fn=generate_data,
        inputs=[
            task_type,
            task_instruction,
            input_instruction,
            output_instruction,
            num_samples,
            llm_provider,
            llm_model,
            llm_api_key,
            llm_base_url,
            task_name,
            domain,
            output_dir,
            cuda_device,
            hf_token,
            language,
            base_model_path,
            semantic_model_path,
            documents,
            dataset_score_threshold,
            answer_extraction_tag,
            answer_extraction_instruction,
            majority_voting_method,
            answer_comparison_method,
            translator_model_path
        ],
        outputs=[generation_status_banner, progress_cards_container, output_files_display, output_section, download_file],
        js="""
        (...args) => {
            const root = document.querySelector('gradio-app')?.shadowRoot || document;

            // 立即标记为已生成，启用顶部 Generate Dataset tab
            const datasetHolder = root.dataset ? root : (root.host || document.querySelector('gradio-app'));
            if (datasetHolder && datasetHolder.dataset) {
                datasetHolder.dataset.generated = 'true';
            }

            // 启用顶部 Generate Dataset tab
            const navGenerate = root.querySelectorAll('[data-nav="generate"]');
            navGenerate.forEach((btn) => {
                btn.classList.remove('disabled');
            });

            // Switch to Generate Dataset navigation tab
            setTimeout(() => {
                if (root && typeof window.setNavMode === 'function') {
                    window.setNavMode('generate');
                }
            }, 50);

            // Trigger output section display via __sdgShowOutput
            if (window.__sdgShowOutput) {
                window.__sdgShowOutput();
            } else {
                setTimeout(() => {
                    const mainLayout = root.querySelector('#main_layout');
                    if (mainLayout) {
                        mainLayout.classList.remove('wizard-mode');
                    }

                    const outputSection = root.querySelector('#output_section');
                    if (outputSection) {
                        setTimeout(() => {
                            outputSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }, 300);
                    }
                }, 100);
            }
            // Return all input arguments unchanged
            return args;
        }
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
