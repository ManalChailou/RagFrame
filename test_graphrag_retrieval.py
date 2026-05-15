from graphRag_system import CosmicGraphRAGSystem

rag = CosmicGraphRAGSystem()

requirements = [
    "When a Registrar selects Modify Professor, the Registrar edits displayed data, C-Reg validates and updates the record, and displays confirmation or error messages."
]

print("\n=== Functional Users Context ===")
print(rag.get_context_for_functional_users(requirements, app_domain="business"))

print("\n=== Functional Processes Context ===")
print(rag.get_context_for_functional_processes(requirements, app_domain="business"))

print("\n=== Data Groups Context ===")
print(rag.get_context_for_data_groups(
    requirements,
    functional_processes=[{"name": "Modify Professor"}],
    app_domain="business"
))

print("\n=== Sub-Processes Context ===")
print(rag.get_context_for_sub_processes(
    requirements,
    functional_processes=[{"name": "Modify Professor"}],
    data_groups=[{"name": "Professor"}],
    app_domain="business"
))

print("\n=== Validation Context ===")
print(rag.get_validation_context())

rag.close()