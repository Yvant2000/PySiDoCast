from setuptools import setup, Extension
from distutils import ccompiler

compiler = ccompiler.get_default_compiler()
print(compiler)


args = []
if compiler == 'msvc':
    args = ["/O2", "/GS-", "/fp:fast", "/std:c++latest", "/Zc:strictStrings-", "/Ob3", "/openmp"]


def main():
    setup(
          name="pysidocast",
          version="1.0.0",
          description="Python Ray Caster for pygame",
          author="Yvant2000",
          author_email="yvant2000@gmail.com",
          ext_modules=[
              Extension(
                  "pysidocast",
                  ["casting.cpp"],
                  extra_compile_args=args
              )
          ]
          )


if __name__ == "__main__":
    main()
