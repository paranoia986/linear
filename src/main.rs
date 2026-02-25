use nalgebra::DMatrix;
use num_rational::Rational64;
use std::collections::HashMap;
use std::io::{self, Write};

// 自定义矩阵打印：美观的带括号和对齐的表格输出
fn print_matrix(mat: &DMatrix<Rational64>) {
    let (rows, cols) = (mat.nrows(), mat.ncols());
    if rows == 0 || cols == 0 {
        println!("[ 空矩阵 ]");
        return;
    }

    // 计算每列的最大宽度，用于格式化对齐
    let mut col_widths = vec![0; cols];
    let mut str_mat = vec![vec![String::new(); cols]; rows];

    for r in 0..rows {
        for c in 0..cols {
            let val = mat[(r, c)];
            let s = if *val.denom() == 1 {
                val.numer().to_string()
            } else {
                format!("{}/{}", val.numer(), val.denom())
            };
            col_widths[c] = col_widths[c].max(s.len());
            str_mat[r][c] = s;
        }
    }

    for row in &str_mat {
        print!("[ ");
        for c in 0..cols {
            print!("{:>width$} ", row[c], width = col_widths[c]);
        }
        println!("]");
    }
}

// 解析输入字符串：支持整数 `3`、分数 `3/4`，以及小数 `1.25`
fn parse_rational(s: &str) -> Result<Rational64, String> {
    // 1. 新增：处理小数的逻辑
    if s.contains('.') {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() == 2 {
            // 将 "1.25" 拼接为 "125"
            let num_str = format!("{}{}", parts[0], parts[1]);
            let num = num_str.parse::<i64>().map_err(|_| format!("无法解析小数: {}", s))?;
            
            // 分母为 10 的 n 次方 (n 为小数位数)
            let decimals = parts[1].len() as u32;
            let den = 10_i64.pow(decimals);
            
            // Rational64::new 会自动把 125/100 约分为最简分数 5/4
            return Ok(Rational64::new(num, den));
        } else {
            return Err(format!("不支持的小数格式: {}", s));
        }
    }

    // 2. 原有的分数和整数处理逻辑
    let parts: Vec<&str> = s.split('/').collect();
    if parts.len() == 1 {
        let num = parts[0].parse::<i64>().map_err(|_| format!("无法解析数字: {}", s))?;
        Ok(Rational64::from_integer(num))
    } else if parts.len() == 2 {
        let num = parts[0].parse::<i64>().map_err(|_| format!("无法解析分子: {}", parts[0]))?;
        let den = parts[1].parse::<i64>().map_err(|_| format!("无法解析分母: {}", parts[1]))?;
        if den == 0 {
            return Err("分母不能为0！".to_string());
        }
        Ok(Rational64::new(num, den))
    } else {
        Err(format!("无法解析分数格式: {}", s))
    }
}

// 辅助函数：将字符串解析为动态分数矩阵
fn parse_matrix(input: &str) -> Result<DMatrix<Rational64>, String> {
    let rows: Vec<&str> = input.trim().split(';').filter(|s| !s.trim().is_empty()).collect();
    let mut data = Vec::new();
    let mut ncols = 0;

    for (i, row) in rows.iter().enumerate() {
        let mut cols = Vec::new();
        for s in row.split_whitespace() {
            // 这里是关键：调用我们自定义的 parse_rational，而不是 f64 的 parse
            cols.push(parse_rational(s)?);
        }

        if i == 0 {
            ncols = cols.len();
        } else if cols.len() != ncols {
            return Err("解析失败：矩阵各行的元素数量不一致！".to_string());
        }
        data.extend(cols);
    }

    let nrows = rows.len();
    if nrows == 0 || ncols == 0 {
        return Err("解析失败：矩阵为空！".to_string());
    }

    Ok(DMatrix::from_row_slice(nrows, ncols, &data))
}

// 尝试从变量表中读取，或者解析为新矩阵
fn read_matrix_or_var(prompt: &str, vars: &HashMap<String, DMatrix<Rational64>>) -> Result<DMatrix<Rational64>, String> {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let input = input.trim();

    if let Some(mat) = vars.get(input) {
        println!("(已加载变量 {})", input);
        Ok(mat.clone())
    } else {
        // 判断输入是否符合变量名的特征：以字母开头，且只包含字母、数字或下划线
        let is_var_name = input.chars().all(|c| c.is_alphanumeric() || c == '_') 
            && input.chars().next().is_some_and(|c| c.is_alphabetic());
        
        if is_var_name {
            // 既然它是个变量名，但没在 HashMap 里找到，就拦截并报错
            return Err(format!("错误：未定义该变量 '{}'。你可以输入 'vars' 查看已定义的变量。", input));
        }

        // 如果不是纯变量名，尝试将其作为普通矩阵字符串解析
        parse_matrix(input)
    }
}

// 核心算法：精确分数高斯-若尔当消元，同时返回 RREF、行列式和秩
fn compute_rref_det_rank(mut mat: DMatrix<Rational64>) -> (DMatrix<Rational64>, Rational64, usize) {
    let mut lead = 0;
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut det = Rational64::from_integer(1);
    let mut rank = 0;
    let mut swaps = 0;

    for r in 0..rows {
        if cols <= lead { break; }
        let mut i = r;
        while *mat[(i, lead)].numer() == 0 {
            i += 1;
            if rows == i {
                i = r;
                lead += 1;
                if cols == lead {
                    let final_det = if rows == cols && rank == rows { det } else { Rational64::from_integer(0) };
                    return (mat, final_det, rank);
                }
            }
        }

        if i != r {
            mat.swap_rows(i, r);
            swaps += 1;
        }

        let val = mat[(r, lead)];
        det *= val;

        for j in 0..cols {
            mat[(r, j)] /= val;
        }

        for i in 0..rows {
            if i != r {
                let val = mat[(i, lead)];
                for j in 0..cols {
                    let sub = val * mat[(r, j)];
                    mat[(i, j)] -= sub;
                }
            }
        }
        lead += 1;
        rank += 1;
    }

    if swaps % 2 != 0 {
        det *= Rational64::from_integer(-1);
    }
    
    if rows != cols || rank < rows {
        det = Rational64::from_integer(0);
    }

    (mat, det, rank)
}

// 通过增广矩阵精确求逆
fn compute_inverse(mat: &DMatrix<Rational64>) -> Option<DMatrix<Rational64>> {
    let n = mat.nrows();
    if n != mat.ncols() { return None; }

    let mut aug = DMatrix::from_element(n, 2 * n, Rational64::from_integer(0));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = mat[(i, j)];
        }
        aug[(i, i + n)] = Rational64::from_integer(1);
    }

    let (rref_aug, _, rank) = compute_rref_det_rank(aug);
    if rank < n {
        return None; 
    }

    let mut inv = DMatrix::from_element(n, n, Rational64::from_integer(0));
    for i in 0..n {
        for j in 0..n {
            inv[(i, j)] = rref_aug[(i, j + n)];
        }
    }
    Some(inv)
}

// 3. 表达式解析器 (AST Parser)
#[derive(Debug, Clone, PartialEq)]
enum Token {
    Plus, Minus, Star, LParen, RParen, Inv, T, Var(String),
}

fn tokenize(input: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let mut chars = input.chars().peekable();

    while let Some(&c) = chars.peek() {
        match c {
            ' ' | '\t' | '\r' | '\n' => { chars.next(); },
            '+' => { tokens.push(Token::Plus); chars.next(); },
            '-' => { tokens.push(Token::Minus); chars.next(); },
            '*' => { tokens.push(Token::Star); chars.next(); },
            '(' => { tokens.push(Token::LParen); chars.next(); },
            ')' => { tokens.push(Token::RParen); chars.next(); },
            _ if c.is_alphabetic() => {
                let mut name = String::new();
                while let Some(&ch) = chars.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        name.push(ch);
                        chars.next();
                    } else { break; }
                }
                match name.as_str() {
                    "inv" => tokens.push(Token::Inv),
                    "t" => tokens.push(Token::T),
                    _ => tokens.push(Token::Var(name)),
                }
            },
            _ => return Err(format!("表达式包含非法字符: {}", c)),
        }
    }
    Ok(tokens)
}

fn parse_factor(tokens: &[Token], pos: &mut usize, vars: &HashMap<String, DMatrix<Rational64>>) -> Result<DMatrix<Rational64>, String> {
    if *pos >= tokens.len() { return Err("表达式不完整".to_string()); }
    let token = &tokens[*pos];
    *pos += 1;

    match token {
        Token::LParen => {
            let mat = parse_expr(tokens, pos, vars)?;
            if *pos >= tokens.len() || tokens[*pos] != Token::RParen { return Err("缺少右括号 ')'".to_string()); }
            *pos += 1;
            Ok(mat)
        },
        Token::Inv => {
            if *pos >= tokens.len() || tokens[*pos] != Token::LParen { return Err("inv 后面必须跟 '('".to_string()); }
            *pos += 1;
            let mat = parse_expr(tokens, pos, vars)?;
            if *pos >= tokens.len() || tokens[*pos] != Token::RParen { return Err("inv() 缺少右括号 ')'".to_string()); }
            *pos += 1;
            compute_inverse(&mat).ok_or_else(|| "错误：矩阵不可逆 (奇异矩阵)".to_string())
        },
        Token::T => {
            if *pos >= tokens.len() || tokens[*pos] != Token::LParen { return Err("t 后面必须跟 '('".to_string()); }
            *pos += 1;
            let mat = parse_expr(tokens, pos, vars)?;
            if *pos >= tokens.len() || tokens[*pos] != Token::RParen { return Err("t() 缺少右括号 ')'".to_string()); }
            *pos += 1;
            Ok(mat.transpose())
        },
        Token::Var(name) => {
            vars.get(name).cloned().ok_or_else(|| format!("未定义变量: {}", name))
        },
        _ => Err(format!("语法错误，意外的符号: {:?}", token)),
    }
}

fn parse_term(tokens: &[Token], pos: &mut usize, vars: &HashMap<String, DMatrix<Rational64>>) -> Result<DMatrix<Rational64>, String> {
    let mut left = parse_factor(tokens, pos, vars)?;
    while *pos < tokens.len() && tokens[*pos] == Token::Star {
        *pos += 1;
        let right = parse_factor(tokens, pos, vars)?;
        if left.ncols() != right.nrows() {
            return Err(format!("乘法维度不匹配: {}x{} * {}x{}", left.nrows(), left.ncols(), right.nrows(), right.ncols()));
        }
        left *= right;
    }
    Ok(left)
}

fn parse_expr(tokens: &[Token], pos: &mut usize, vars: &HashMap<String, DMatrix<Rational64>>) -> Result<DMatrix<Rational64>, String> {
    let mut left = parse_term(tokens, pos, vars)?;
    while *pos < tokens.len() && (tokens[*pos] == Token::Plus || tokens[*pos] == Token::Minus) {
        let is_plus = tokens[*pos] == Token::Plus;
        *pos += 1;
        let right = parse_term(tokens, pos, vars)?;
        if left.nrows() != right.nrows() || left.ncols() != right.ncols() {
            return Err(format!("加减法维度不匹配: {}x{} 和 {}x{}", left.nrows(), left.ncols(), right.nrows(), right.ncols()));
        }
        if is_plus { left += right; } else { left -= right; }
    }
    Ok(left)
}

fn eval_expression(expr: &str, vars: &HashMap<String, DMatrix<Rational64>>) -> Result<DMatrix<Rational64>, String> {
    let tokens = tokenize(expr)?;
    let mut pos = 0;
    let result = parse_expr(&tokens, &mut pos, vars)?;
    if pos < tokens.len() {
        return Err("表达式解析未完全，请检查语法".to_string());
    }
    Ok(result)
}

fn main() {
    println!("=======================================================");
    println!("  Matrix计算器");
    println!("  支持指令:");
    println!("  - let <var> = <mat> : 定义变量 (例如: let A = 1 2; 3/4 5)");
    println!("  - vars              : 查看所有已定义的变量");
    println!("  - det        : 行列式");
    println!("  - t          : 转置");
    println!("  - add        : 矩阵相加");
    println!("  - mul        : 乘法");
    println!("  - inv        : 逆矩阵");
    println!("  - rank       : 求秩");
    println!("  - rref       : 最简行阶梯矩阵");
    println!("  - eigen      : 特征值与特征向量 (专为实对称矩阵优化)");
    println!("  - cal        : 混合表达式求值 (例如: inv(A) * B + t(C))");
    println!("  - quit/exit  : 退出");
    println!("  提示: 遇到输入矩阵的提示时，可直接输入矩阵或输入变量名(如 A)");
    println!("  矩阵格式举例: 1 2 3; 4 5 6; 7 8 9");
    println!("=======================================================");

    let mut variables: HashMap<String, DMatrix<Rational64>> = HashMap::new();

    loop {
        print!("\n> 请输入指令: ");
        io::stdout().flush().unwrap();
        
        let mut raw_command = String::new();
        io::stdin().read_line(&mut raw_command).unwrap();
        
        // 保留原始大小写的字符串，用于提取变量名
        let command = raw_command.trim();
        // 专门生成一个全小写的版本，只用于指令判断
        let cmd_lower = command.to_lowercase();

        if cmd_lower == "quit" || cmd_lower == "exit" {
            println!("再见！");
            break;
        }

        // 处理变量定义 let A = ...
        if cmd_lower.starts_with("let ") {
            // 使用原字符串 command 进行截取，保留变量名的大小写
            let rest = &command[4..]; 
            let parts: Vec<&str> = rest.splitn(2, '=').collect();
            if parts.len() == 2 {
                let var_name = parts[0].trim().to_string(); // 这里完美保留了大小写
                let mat_str = parts[1].trim();
                match parse_matrix(mat_str) {
                    Ok(mat) => {
                        println!("变量 {} 保存成功:", var_name);
                        print_matrix(&mat);
                        variables.insert(var_name, mat);
                    }
                    Err(e) => println!("{}", e),
                }
            } else {
                println!("格式错误。请使用: let A = 1 2; 3 4");
            }
            continue;
        }

        if cmd_lower == "vars" {
            if variables.is_empty() {
                println!("当前没有保存的变量。");
            } else {
                for (name, mat) in &variables {
                    println!("变量 {}:", name);
                    print_matrix(mat);
                    println!();
                }
            }
            continue;
        }

        // 需要读取单矩阵的指令集合
        if ["det", "t", "inv", "rank", "rref", "eigen"].contains(&cmd_lower.as_str()) {
            match read_matrix_or_var("> 请输入矩阵 (直接输入或填变量名): ", &variables) {
                Ok(mat) => {
                    match cmd_lower.as_str() {
                        "det" => {
                            if mat.is_square() {
                                let (_, det, _) = compute_rref_det_rank(mat);
                                let det_str = if *det.denom() == 1 { det.numer().to_string() } else { format!("{}/{}", det.numer(), det.denom()) };
                                println!("行列式 |A| = {}", det_str);
                            } else {
                                println!("错误：只有方阵才能计算行列式！");
                            }
                        }
                        "t" => {
                            println!("转置矩阵 A^T:");
                            print_matrix(&mat.transpose());
                        }
                        "inv" => {
                            if mat.is_square() {
                                match compute_inverse(&mat) {
                                    Some(inv) => {
                                        println!("逆矩阵 A^-1:");
                                        print_matrix(&inv);
                                    }
                                    None => println!("结果：该矩阵不可逆 (奇异矩阵，|A|=0)！"),
                                }
                            } else {
                                println!("错误：只有方阵才有逆矩阵！");
                            }
                        }
                        "rank" => {
                            let (_, _, rank) = compute_rref_det_rank(mat);
                            println!("矩阵的秩 Rank(A) = {}", rank);
                        }
                        "rref" => {
                            let (rref_mat, _, _) = compute_rref_det_rank(mat);
                            println!("最简行阶梯矩阵 (RREF):");
                            print_matrix(&rref_mat);
                        }
                        "eigen" => {
                            if mat.is_square() {
                                // 将分数转换为 f64 进行特征值计算（特征值经常涉及无理数）
                                let mut f64_mat = DMatrix::from_element(mat.nrows(), mat.ncols(), 0.0);
                                for i in 0..mat.nrows() {
                                    for j in 0..mat.ncols() {
                                        f64_mat[(i, j)] = *mat[(i, j)].numer() as f64 / *mat[(i, j)].denom() as f64;
                                    }
                                }

                                let diff = f64_mat.clone() - f64_mat.transpose();
                                let is_symmetric = diff.iter().all(|&x| x.abs() < 1e-7);

                                if is_symmetric {
                                    let eigen = nalgebra::linalg::SymmetricEigen::new(f64_mat);
                                    println!("检测到实对称矩阵，特征值与特征向量如下 (保留小数)：\n");
                                    println!("特征值:\n{:.4}", eigen.eigenvalues);
                                    println!("正交特征向量矩阵:\n{:.4}", eigen.eigenvectors);
                                } else {
                                    println!("提示：当前矩阵不是实对称矩阵，特征系统可能包含复数，暂不支持。");
                                }
                            } else {
                                println!("错误：只有方阵才能求特征值！");
                            }
                        }
                        _ => {}
                    }
                }
                Err(e) => println!("{}", e),
            }
        } else if cmd_lower == "mul" {
            // 合并判断，同时读取 A 和 B（注意：Rust 1.65+ 支持这种写法，或者直接通过 match / 元组处理）
            if let (Ok(mat_a), Ok(mat_b)) = (
                read_matrix_or_var("> 请输入矩阵 A: ", &variables),
                read_matrix_or_var("> 请输入矩阵 B: ", &variables)
            ) {
                if mat_a.ncols() == mat_b.nrows() {
                    let result = mat_a * mat_b;
                    println!("乘积 AB = :");
                    print_matrix(&result);
                } else {
                    println!("错误：矩阵 A 的列数 ({}) 必须等于矩阵 B 的行数 ({})！", mat_a.ncols(), mat_b.nrows());
                }
               }
            }else if cmd_lower == "add" {
            // 独立 add 指令
            if let (Ok(mat_a), Ok(mat_b)) = (read_matrix_or_var("> 请输入矩阵 A: ", &variables), read_matrix_or_var("> 请输入矩阵 B: ", &variables)) {
                if mat_a.nrows() == mat_b.nrows() && mat_a.ncols() == mat_b.ncols() {
                    let result = mat_a + mat_b;
                    println!("和 A + B = :");
                    print_matrix(&result);
                } else { println!("错误：矩阵加法要求同型矩阵！A是{}x{}, B是{}x{}", mat_a.nrows(), mat_a.ncols(), mat_b.nrows(), mat_b.ncols()); }
            }
            } else if cmd_lower == "cal" {
            print!("> 请输入表达式 (例如: inv(A) * B + t(C)): ");
            io::stdout().flush().unwrap();
            let mut expr = String::new();
            io::stdin().read_line(&mut expr).unwrap();
            let expr = expr.trim();

            match eval_expression(expr, &variables) {
                Ok(result) => {
                    println!("表达式 \"{}\" 的结果为:", expr);
                    print_matrix(&result);
                }
                Err(e) => println!("表达式计算错误: {}", e),
            }
        } else if !command.is_empty() {
            // 当输入了未能识别的指令时提示
            println!("未知指令，请参考启动时的提示列表。");
        }
    }
}