from setuptools import setup, Extension
from distutils import ccompiler

compiler = ccompiler.get_default_compiler()
# print(compiler)


args = []
if compiler == 'msvc':
    args = ["/O2", "/GS-", "/fp:fast", "/std:c++latest", "/Zc:strictStrings-", "/Ob3"]  # , "/Wall"


def main():
    setup(
        name="pysidocast",
        version="1.3.0",
        description="Python Ray Caster for pygame",
        author="Yvant2000",
        author_email="yvant2000@gmail.com",
        ext_modules=[
            Extension(
                "pysidocast",
                ["src/casting.cpp"],
                include_dirs=["src/"],
                language="c++",
                extra_compile_args=args
            )
        ]
    )


if __name__ == "__main__":
    main()
