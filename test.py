import ast



code = """
def hello_world(name: str) -> str:
    print(f"Hello, " + name)

hello_world(name)
"""

# Execute code with redirected stdout and stderr
exec(code, {'name': 'world'})
# Safely evaluate and retrieve result
# exec_result = ast.literal_eval(repr(context.get('result', True)))
# print("Exec result:", exec_result)
no_runtime_error = True