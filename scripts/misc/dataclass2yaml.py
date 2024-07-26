
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
from dataclasses import is_dataclass, fields
from run_config import RunConfig

def dataclass_to_yaml_dict_recursive(dataclass_instance):
    yaml_dict = CommentedMap()
    if is_dataclass(dataclass_instance):
        for field in fields(dataclass_instance):
            field_name = field.name
            field_default = getattr(dataclass_instance, field.name)
            field_help = field.metadata.get("help", None)

            if is_dataclass(field_default):
                yaml_dict[field_name] = dataclass_to_yaml_dict_recursive(field_default)
            else:
                yaml_dict[field_name] = field_default
            yaml_dict.yaml_set_comment_before_after_key(field_name, before=field_help, indent=2)
    else:
        raise ValueError("Input must be a dataclass")

    return yaml_dict


if __name__ == "__main__":
    run_config_instance = RunConfig()
    yaml_dict = dataclass_to_yaml_dict_recursive(run_config_instance)

    # Write the YAML string to a file
    yaml = YAML()
    with open('run_config2.yaml', 'w', encoding='utf-8') as yaml_file:
        yaml.dump(yaml_dict, yaml_file)

    # print(yaml_dict)
