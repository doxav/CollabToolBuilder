import re

# Read HTML file
with open("Jquery_front/IHMv5-Monaco.html", "r", encoding="utf-8") as file:
    html = file.read()

# Inline CSS
with open("Jquery_front/style.css", "r", encoding="utf-8") as css_file:
    css_content = css_file.read()
html = re.sub(r'<link rel="stylesheet" href="style.css">', f'<style>{css_content}</style>', html)

# List of JavaScript files to inline
js_files = [
    "Jquery_front/global.js",
    "Jquery_front/utils.js",
    "Jquery_front/display-message-helpers.js",
    "Jquery_front/display-message.js",
    "Jquery_front/websocket-handler.js",
    "Jquery_front/floating-div.js",
    "Jquery_front/monaco-log-ui-manager.js"
]

# Inline JavaScript safely
for js_file in js_files:
    with open(js_file, "r", encoding="utf-8") as js:
        js_content = js.read()

    # Fix escaping issue: Only escape backslashes, but keep $ intact
    safe_js_content = js_content.replace('\\', '\\\\')

    # Replace <script src="..."> with inline script
    html = re.sub(fr'<script\s+src="{re.escape(js_file.split("/")[-1])}"></script>',
                  f'<script>\n{safe_js_content}\n</script>',
                  html,
                  flags=re.IGNORECASE)

# Save the modified file
with open("Jquery_front/Monaco-Editor-Bundled.html", "w", encoding="utf-8") as output_file:
    output_file.write(html)

print("Inlining completed. Now run html-minifier to minify the file.")