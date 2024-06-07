import collections
import re
from io import StringIO
import tokenize


def isSameTree(root_p, root_q) -> bool:
    if not root_p and not root_q:
        return True
    if not root_p or not root_q:
        return False
    queue_p = collections.deque([root_p])
    queue_q = collections.deque([root_q])
    while queue_p and queue_q:
        node_p = queue_p.popleft()
        node_q = queue_q.popleft()
        if node_p.type != node_q.type:
            return False
        if len(node_p.children) != len(node_q.children):
            return False
        if len(node_p.children) > 0:
            for child_p, child_q in zip(node_p.children, node_q.children) :
                if child_p.type == child_q.type:
                    queue_p.append(child_p)
                    queue_p.append(child_q)
                else:
                    return False
    return True


def remove_comments_and_docstrings(source,lang):
    if lang in ['python']:
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            if token_type == tokenize.COMMENT:
                pass
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['ruby']:
        return source
    else:
        def replacer(match):
            s = match.group(0)
            if s.startswith('/'):
                return " "
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)


def tree_to_token_index(root_node):
    if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
        return code_tokens


def tree_to_variable_index(root_node,index_to_code):
    if root_node:
        if (len(root_node.children) == 0 or root_node.type == 'string') and root_node.type != 'comment':
            index = (root_node.start_point, root_node.end_point)
            _, code = index_to_code[index]
            if root_node.type != code:
                return [(root_node.start_point, root_node.end_point)]
            else:
                return []
        else:
            code_tokens = []
            for child in root_node.children:
                code_tokens += tree_to_variable_index(child, index_to_code)
            return code_tokens  
    else:
        return []


def index_to_code_token(index, code):
    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s


def tree_to_variable_poses(tree, code, code_tokens, lang='c_sharp'):
    def _traverse_tree(tree):
        cursor = tree.walk()
        reached_root = False
        while reached_root == False:
            yield cursor.node
            if cursor.goto_first_child():
                continue
            if cursor.goto_next_sibling():
                continue
            retracing = True
            while retracing:
                if not cursor.goto_parent():
                    retracing = False
                    reached_root = True
                if cursor.goto_next_sibling():
                    retracing = False

    def _convert_rowcol_to_idx(start_point, end_point, code):
        row_len_list = [len(r) for r in code.split('\n')]
        for i in range(0, len(row_len_list)): row_len_list[i] += 1 # count the \n in each row
        accumulated_len_list = [0] + [sum(row_len_list[:i]) for i in range(1, len(row_len_list)+1)]
        row_idx, col_idx = start_point
        start_offset = accumulated_len_list[row_idx] + col_idx
        row_idx, col_idx = end_point
        end_offset = accumulated_len_list[row_idx] + col_idx
        return start_offset, end_offset

    tlist = ['void', 'const', 'inline', 'int', 'float', 'double', 'static'] +\
         ['uint{}_t'.format(x) for x in [8, 16, 32, 64]] + ['int{}_t'.format(x) for x in [8, 16, 32, 64]]
    declare_lsit = ['assignment_expression', 'local_function_statement',\
         'global_statement', 'identifier', 'variable_declarator']
    variable_set = []
    is_last_dec = False
    for node in _traverse_tree(tree):
        s, e = _convert_rowcol_to_idx(node.start_point, node.end_point, code)
        cur_segment = code[s: e]
        if cur_segment == 'self' and lang == 'python':
            continue
        if node.type == 'identifier' and node.end_point[0] == 0\
             and cur_segment.lower() != 'override' and cur_segment not in tlist:
            if cur_segment not in variable_set:
                variable_set.append(cur_segment) # func names
        if is_last_dec and node.type == 'identifier' and cur_segment not in tlist:
            if cur_segment not in variable_set:
                variable_set.append(cur_segment)
        is_last_dec = node.type in declare_lsit
    variable_poses = []
    tag_map = {v: i for i, v in enumerate(variable_set)}
    for token in code_tokens:
        if token in variable_set:
            variable_poses.append(tag_map[token])
        else:
            variable_poses.append(0) # non-entity token
    # print("variable_set:", variable_set)
    # print("variable_poses:", variable_poses)
    return variable_poses

