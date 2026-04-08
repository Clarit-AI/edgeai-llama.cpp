#include "chat-peg-parser.h"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

static std::string escape_json_string_local(const std::string & value) {
    std::string dumped = json(value).dump();
    return dumped.size() >= 2 ? dumped.substr(1, dumped.size() - 2) : dumped;
}

static common_peg_ast_id find_child_by_tag(const common_peg_ast_arena & arena, const common_peg_ast_node & node, const std::string & tag) {
    for (auto child_id : node.children) {
        const auto & child = arena.get(child_id);
        if (child.tag == tag) {
            return child_id;
        }
        auto nested = find_child_by_tag(arena, child, tag);
        if (nested != COMMON_PEG_INVALID_AST_ID) {
            return nested;
        }
    }
    return COMMON_PEG_INVALID_AST_ID;
}

static std::string_view trim_trailing_space(std::string_view sv, int max = -1) {
    int count = 0;
    while (!sv.empty() && std::isspace(static_cast<unsigned char>(sv.back()))) {
        if (max != -1 && count <= max) {
            break;
        }
        sv.remove_suffix(1);
        count++;
    }
    return sv;
}

void common_chat_peg_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    arena.visit(result, [this](const common_peg_ast_node & node) {
        map(node);
    });
}

void common_chat_peg_mapper::map(const common_peg_ast_node & node) {
    bool is_reasoning = node.tag == common_chat_peg_builder::REASONING;
    bool is_content = node.tag == common_chat_peg_builder::CONTENT;

    if (is_reasoning) {
        result.reasoning_content = std::string(trim_trailing_space(node.text));
    }

    if (is_content) {
        result.content = std::string(trim_trailing_space(node.text));
    }
}

void common_chat_peg_gemma4_mapper::from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result) {
    for (const auto & node : result.nodes) {
        visit(arena, node);
    }
}

static std::string gemma4_to_json(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.text.empty()) {
        return "";
    }

    if (node.rule == "gemma4-number" || node.rule == "gemma4-bool" || node.rule == "gemma4-null") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-string-content") {
        return escape_json_string_local(std::string(node.text));
    }

    if (node.rule == "gemma4-string") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += gemma4_to_json(arena, node.children[0]);
            if (!node.is_partial) {
                result += "\"";
            }
        }
        return result;
    }

    if (node.rule == "gemma4-array") {
        std::string result = "[";
        bool add_comma = false;

        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += ']';
        }
        return result;
    }

    if (node.rule == "gemma4-dict-key-name") {
        return std::string(node.text);
    }

    if (node.rule == "gemma4-dict-key") {
        std::string result = "\"";
        if (!node.children.empty()) {
            result += escape_json_string_local(gemma4_to_json(arena, node.children[0]));
        }
        if (!node.is_partial) {
            result += "\":";
        }
        return result;
    }

    if (node.rule == "gemma4-dict-kv") {
        std::string result;
        for (auto child_id : node.children) {
            result += gemma4_to_json(arena, child_id);
        }
        return result;
    }

    if (node.rule == "gemma4-dict") {
        std::string result = "{";
        bool add_comma = false;

        for (auto child_id : node.children) {
            if (add_comma) {
                result += ',';
            }
            add_comma = true;
            result += gemma4_to_json(arena, child_id);
        }

        if (!node.is_partial) {
            result += '}';
        }
        return result;
    }

    if (node.rule == "gemma4-value") {
        if (!node.children.empty()) {
            return gemma4_to_json(arena, node.children[0]);
        }
        return "";
    }

    return "";
}

void common_chat_peg_gemma4_mapper::visit(const common_peg_ast_arena & arena, common_peg_ast_id id) {
    const auto & node = arena.get(id);

    if (node.tag == common_chat_peg_builder::REASONING) {
        result.reasoning_content += std::string(node.text);
        return;
    }

    if (node.tag == common_chat_peg_builder::CONTENT) {
        result.content += std::string(node.text);
        return;
    }

    if (node.tag == common_chat_peg_native_builder::TOOL) {
        auto name_id = find_child_by_tag(arena, node, common_chat_peg_native_builder::TOOL_NAME);
        auto args_id = find_child_by_tag(arena, node, common_chat_peg_native_builder::TOOL_ARGS);

        if (name_id != COMMON_PEG_INVALID_AST_ID && args_id != COMMON_PEG_INVALID_AST_ID) {
            const auto & name_node = arena.get(name_id);
            const auto & args_node = arena.get(args_id);

            if (!name_node.is_partial) {
                common_chat_tool_call call;
                call.name = std::string(name_node.text);
                if (!args_node.children.empty()) {
                    call.arguments = gemma4_to_json(arena, args_node.children[0]);
                }
                result.tool_calls.push_back(call);
            }
        }

        return;
    }

    for (auto child_id : node.children) {
        visit(arena, child_id);
    }
}

void common_chat_peg_native_mapper::map(const common_peg_ast_node & node) {
    common_chat_peg_mapper::map(node);

    bool is_tool_open = node.tag == common_chat_peg_native_builder::TOOL_OPEN;
    bool is_tool_name = node.tag == common_chat_peg_native_builder::TOOL_NAME;
    bool is_tool_id = node.tag == common_chat_peg_native_builder::TOOL_ID;
    bool is_tool_args = node.tag == common_chat_peg_native_builder::TOOL_ARGS;

    if (is_tool_open) {
        result.tool_calls.emplace_back();
        current_tool = &result.tool_calls.back();
    }

    if (is_tool_id && current_tool) {
        current_tool->id = std::string(trim_trailing_space(node.text));
    }

    if (is_tool_name && current_tool) {
        current_tool->name = std::string(trim_trailing_space(node.text));
    }

    if (is_tool_args && current_tool) {
        current_tool->arguments = std::string(trim_trailing_space(node.text));
    }
}

void common_chat_peg_constructed_mapper::map(const common_peg_ast_node & node) {
    common_chat_peg_mapper::map(node);

    bool is_tool_open = node.tag == common_chat_peg_constructed_builder::TOOL_OPEN;
    bool is_tool_name = node.tag == common_chat_peg_constructed_builder::TOOL_NAME;
    bool is_tool_close = node.tag == common_chat_peg_constructed_builder::TOOL_CLOSE;
    bool is_arg_open = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_OPEN;
    bool is_arg_close = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_CLOSE;
    bool is_arg_name = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_NAME;
    bool is_arg_string = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_STRING_VALUE;
    bool is_arg_json = node.tag == common_chat_peg_constructed_builder::TOOL_ARG_JSON_VALUE;

    if (is_tool_open) {
        result.tool_calls.emplace_back();
        current_tool = &result.tool_calls.back();
        arg_count = 0;
    }

    if (is_tool_name) {
        current_tool->name = std::string(node.text);
        current_tool->arguments = "{";
    }

    if (is_arg_open) {
        needs_closing_quote = false;
    }

    if (is_arg_name && current_tool) {
        if (arg_count > 0) {
            current_tool->arguments += ",";
        }
        current_tool->arguments += json(trim_trailing_space(node.text)).dump() + ":";
        ++arg_count;
    }

    if (is_arg_string && current_tool) {
        // Serialize to JSON, but exclude the end quote
        std::string dumped = json(trim_trailing_space(node.text)).dump();
        current_tool->arguments += dumped.substr(0, dumped.size() - 1);
        needs_closing_quote = true;
    }

    if (is_arg_close && current_tool) {
        if (needs_closing_quote) {
            current_tool->arguments += "\"";
            needs_closing_quote = false;
        }
    }

    if (is_arg_json && current_tool) {
        current_tool->arguments += std::string(trim_trailing_space(node.text));
    }

    if (is_tool_close && current_tool) {
        if (needs_closing_quote) {
            current_tool->arguments += "\"";
            needs_closing_quote = false;
        }
        current_tool->arguments += "}";
    }
}
