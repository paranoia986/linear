use nalgebra::DMatrix;
use std::io::{self, Write};

// 辅助函数：将字符串解析为动态矩阵
fn parse_matrix(input: &str) -> Result<DMatrix<f64>, String> {
    let rows: Vec<&str> = input.trim().split(';').filter(|s| !s.trim().is_empty()).collect();
    let mut data = Vec::new();
    let mut ncols = 0;

    for (i, row) in rows.iter().enumerate() {
        let cols: Vec<f64> = row
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        
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

// 核心算法：高斯-若尔当消元法求最简行阶梯矩阵 (RREF)
fn compute_rref(mut mat: DMatrix<f64>) -> DMatrix<f64> {
    let mut lead = 0;
    let rows = mat.nrows();
    let cols = mat.ncols();

    for r in 0..rows {
        if cols <= lead {
            break;
        }
        let mut i = r;
        while mat[(i, lead)].abs() < 1e-10 {
            i += 1;
            if rows == i {
                i = r;
                lead += 1;
                if cols == lead {
                    return mat;
                }
            }
        }
        
        // 列主元选取 (Partial Pivoting) 以保证数值稳定
        let mut max_row = i;
        for k in i..rows {
            if mat[(k, lead)].abs() > mat[(max_row, lead)].abs() {
                max_row = k;
            }
        }
        mat.swap_rows(max_row, r);

        // 归一化主元行
        let val = mat[(r, lead)];
        if val.abs() > 1e-10 {
            for j in 0..cols {
                mat[(r, j)] /= val;
            }
        }
        
        // 消元其他行
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
    }

    // 清理非常小的浮点数误差 (如 -0.0) 以保持输出美观
    for x in mat.iter_mut() {
        if x.abs() < 1e-10 {
            *x = 0.0;
        }
    }
    mat
}

fn main() {
    println!("=======================================================");
    println!("  Matrix计算器");
    println!("  支持指令:");
    println!("  - det   : 行列式");
    println!("  - t     : 转置");
    println!("  - mul   : 乘法");
    println!("  - inv   : 逆矩阵");
    println!("  - rank  : 求秩");
    println!("  - rref  : 最简行阶梯矩阵");
    println!("  - eigen : 特征值与特征向量 (专为实对称矩阵优化)");
    println!("  - quit  : 退出");
    println!("  矩阵格式举例: 1 2 3; 4 5 6; 7 8 9");
    println!("=======================================================");

    loop {
        print!("\n> 请输入指令: ");
        io::stdout().flush().unwrap();
        
        let mut command = String::new();
        io::stdin().read_line(&mut command).unwrap();
        let command = command.trim().to_lowercase();

        if command == "quit" || command == "exit" {
            println!("再见！");
            break;
        }

        // 需要读取单矩阵的指令集合
        if ["det", "t", "inv", "rank", "rref", "eigen"].contains(&command.as_str()) {
            print!("> 请输入矩阵: ");
            io::stdout().flush().unwrap();
            let mut mat_str = String::new();
            io::stdin().read_line(&mut mat_str).unwrap();

            match parse_matrix(&mat_str) {
                Ok(mat) => {
                    match command.as_str() {
                        "det" => {
                            if mat.is_square() {
                                println!("行列式 |A| = {:.4}", mat.determinant());
                            } else {
                                println!("错误：只有方阵才能计算行列式！");
                            }
                        }
                        "t" => {
                            println!("转置矩阵 A^T:\n{:.4}", mat.transpose());
                        }
                        "inv" => {
                            if mat.is_square() {
                                match mat.clone().try_inverse() {
                                    Some(inv) => println!("逆矩阵 A^-1:\n{:.4}", inv),
                                    None => println!("结果：该矩阵不可逆 (奇异矩阵，|A|=0)！"),
                                }
                            } else {
                                println!("错误：只有方阵才有逆矩阵！");
                            }
                        }
                        "rank" => {
                            // nalgebra 原生求秩，传入误差容限
                            let r = mat.rank(1e-7);
                            println!("矩阵的秩 Rank(A) = {}", r);
                        }
                        "rref" => {
                            let rref_mat = compute_rref(mat.clone());
                            println!("最简行阶梯矩阵 (RREF):\n{:.4}", rref_mat);
                        }
                        "eigen" => {
                            if mat.is_square() {
                                // 检查是否为实对称矩阵 A = A^T
                                let diff = mat.clone() - mat.transpose();
                                let is_symmetric = diff.iter().all(|&x| x.abs() < 1e-7);

                                if is_symmetric {
                                    let eigen = nalgebra::linalg::SymmetricEigen::new(mat);
                                    println!("✔ 检测到实对称矩阵，一定存在实特征值与正交特征向量！\n");
                                    println!("特征值 (λ1, λ2...):\n{:.4}", eigen.eigenvalues);
                                    println!("特征向量 (每一列对应一个特征值的特征向量):\n{:.4}", eigen.eigenvectors);
                                } else {
                                    println!("提示：当前矩阵不是实对称矩阵。");
                                    println!("不对称矩阵的特征值可能包含复数。为了程序稳定，本工具暂时只计算实对称矩阵的特征系统。");
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
        } else if command == "mul" {
            // 乘法需要读取两个矩阵
            print!("> 请输入矩阵 A: ");
            io::stdout().flush().unwrap();
            let mut a_str = String::new();
            io::stdin().read_line(&mut a_str).unwrap();

            print!("> 请输入矩阵 B: ");
            io::stdout().flush().unwrap();
            let mut b_str = String::new();
            io::stdin().read_line(&mut b_str).unwrap();

            if let (Ok(mat_a), Ok(mat_b)) = (parse_matrix(&a_str), parse_matrix(&b_str)) {
                if mat_a.ncols() == mat_b.nrows() {
                    let result = mat_a * mat_b;
                    println!("\n乘积 AB = :\n{:.4}", result);
                } else {
                    println!("错误：矩阵 A 的列数 ({}) 必须等于矩阵 B 的行数 ({})！", mat_a.ncols(), mat_b.nrows());
                }
            } else {
                println!("错误：矩阵解析失败，请检查输入格式。");
            }
        } else {
            println!("未知指令，请参考启动时的提示列表。");
        }
    }
}