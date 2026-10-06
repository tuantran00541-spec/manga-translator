# probe: minimal plugin to test which tool names are reserved.
inject = ["tools"]
defaults = {}
def apply(ctx, config):
    for name in ["make_pptx", "check_pptx", "build_pptx", "audit_pptx"]:
        try:
            ctx.get("tools").register(dict(name=name, description="probe " + name), lambda s, a, n=name: n, "read")
            print("ok:", name)
        except Exception as e:
            print("fail:", name, e)
