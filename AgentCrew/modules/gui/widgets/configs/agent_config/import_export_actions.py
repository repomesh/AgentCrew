def determine_file_format_and_path(
    file_path: str, selected_filter: str
) -> tuple[str, str]:
    if file_path.lower().endswith(".toml"):
        return file_path, "toml"
    if file_path.lower().endswith(".json"):
        return file_path, "json"

    if "toml" in selected_filter.lower():
        return file_path + ".toml", "toml"
    return file_path + ".json", "json"
