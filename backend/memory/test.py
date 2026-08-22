from backend.memory.lang_memory import MemoryType


def all_types():
    print("all_types")
    print("=" * 60)

    print(MemoryType.all_types())

    print("=" * 60)


def all_values():
    print("all_values")
    print("=" * 60)

    print(MemoryType.all_values())

    print("=" * 60)


# def get_description():
#     print("get_description")
#     print("=" * 60)

#     print(MemoryType.get_description())

#     print("=" * 60)


def get_llm_schema():
    print("get_llm_schema")
    print("=" * 60)

    print(MemoryType.get_llm_schema())

    print("=" * 60)


def get_llm_descriptions():
    print("get_llm_descriptions")
    print("=" * 60)

    print(MemoryType.get_llm_descriptions())

    print("=" * 60)


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    all_types()
    all_values()
    # get_description()
    get_llm_schema()
    get_llm_descriptions()