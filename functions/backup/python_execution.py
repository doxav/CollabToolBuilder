def execute_python_code(code_to_execute: str) -> str:
    """ Executes the given Python code string and returns the output.
    :param code_to_execute: The Python code string to execute
    :return: The output of the Python code
    """

    import subprocess
    from tempfile import NamedTemporaryFile

    with NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(code_to_execute.encode('utf-8'))
        temp_file.flush()
    try:
        result = subprocess.run(
            ['python', temp_file_name],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return e.stderr
    finally:
        import os
        os.remove(temp_file_name)

