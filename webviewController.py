import webview

window = webview.create_window(title='Hygino AI Solution',
                               url='http://localhost:5001',
                               width=1280,
                               height=720,
                               resizable=False,
                               background_color='#171823',
                               text_select=False,
                               on_top=True,
                               frameless=False)

def webView_start():
    webview.start(gui="edgechromium", debug=True)