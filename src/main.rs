use std::ops::{Index,IndexMut};
use std::time::{Instant, Duration};
use std::process;
use std::env;
use std::vec;

#[derive(Debug, PartialEq)]
enum CalculationMethod
{
    MethGaussSeidel,
    MethJacobi,
}

impl std::str::FromStr for CalculationMethod
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err>
    {
        match s
        {
            "MethGaussSeidel" => Ok(CalculationMethod::MethGaussSeidel),
            "MethJacobi" => Ok(CalculationMethod::MethJacobi),
            _ => Err(format!("'{}' is not a valid value for CalculationMethod", s)),
        }
    }
}

#[derive(Debug, PartialEq)]
enum InferenceFunction
{
    FuncF0,
    FuncFPiSin,
}

impl std::str::FromStr for InferenceFunction
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err>
    {
        match s
        {
            "FuncF0" => Ok(InferenceFunction::FuncF0),
            "FuncFPiSin" => Ok(InferenceFunction::FuncFPiSin),
            _ => Err(format!("'{}' is not a valid value for InferenceFunction", s)),
        }
    }
}

#[derive(Debug, PartialEq)]
enum TerminationCondition
{
    TermPrec,
    TermIter,
}

impl std::str::FromStr for TerminationCondition
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err>
    {
        match s
        {
            "TermPrec" => Ok(TerminationCondition::TermPrec),
            "TermIter" => Ok(TerminationCondition::TermIter),
            _ => Err(format!("'{}' is not a valid value for TerminationCondition", s)),
        }
    }
}

#[derive(Debug)]
struct CalculationOptions
{
    number: u64,                        // number of threads
    method: CalculationMethod,                 // Gauss Seidel or Jacobi method of iteration
    interlines: usize,                  // matrix size = interline*8+9
    inf_func: InferenceFunction,        // inference function
    termination: TerminationCondition,  // termination condition
    term_iteration: u64,                // terminate if iteration number reached
    term_precision: f64,                // terminate if precision reached
}

impl CalculationOptions
{
    fn new(number: u64, method: CalculationMethod, interlines: usize, inf_func: InferenceFunction,
        termination: TerminationCondition, term_iteration: u64, term_precision: f64)
        -> CalculationOptions
    {
        CalculationOptions{number, method, interlines, inf_func, termination, term_iteration, term_precision}
    }
}

#[derive(Debug)]
struct CalculationArguments
{
    n: usize,                       // Number of spaces between lines (lines=n+1)
    num_matrices: usize,              // number of matrices
    h: f64,                         // length of a space between two lines
    matrices: Vec<PartdiffMatrix>,  // The matrices for calculation
}

impl CalculationArguments
{
    fn new(n: usize, num_matrices: usize, h: f64) -> CalculationArguments
    {
        let mut matrices: Vec<PartdiffMatrix> = Vec::with_capacity(num_matrices);

        for _ in 0..num_matrices
        {
            let matrix = PartdiffMatrix::new(n+1);
            matrices.push(matrix);
        }

        CalculationArguments{n, num_matrices, h, matrices}
    }
}

#[derive(Debug)]
struct CalculationResults
{
    m: usize,
    stat_iteration: u64,  // number of current iteration
    stat_precision: f64,  // actual precision of all slaces in iteration
}

impl CalculationResults
{
    fn new(m: usize, stat_iteration: u64, stat_precision: f64) -> CalculationResults
    {
        CalculationResults{m, stat_iteration, stat_precision}
    }
}

#[derive(Debug)]
struct PartdiffMatrix
{
    n: usize,
    matrix: Vec<f64>,
}

impl PartdiffMatrix
{
    fn new(n: usize) -> PartdiffMatrix
    {
        let matrix = vec![0.0; ((n+1)*(n+1)) as usize];
        PartdiffMatrix{n, matrix}
    }
}

// impl Index<[usize; 2]> for PartdiffMatrix
// {
//     type Output = f64;
//     
//     // fn index(&self, idx: [usize; 2]) -> &Self::Output
//     // {
//     //     &self.matrix[idx[0] * self.n + idx[1]]
//     // }
//     fn index(&self, idx: [usize; 2]) -> &Self::Output
//     {
//         &self.matrix[idx.0 * self.n + idx.1]
//     }
// }
//
// impl IndexMut<[usize; 2]> for PartdiffMatrix
// {
//     // fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output
//     // {
//     //     &mut self.matrix[idx[0] * self.n + idx[1]]
//     // }
//     fn index_mut(&mut self, idx: i[usize; 2]) -> &mut Self::Output
//     {
//         &mut self.matrix[idx.0 * self.n + idx.1]
//     }
// }

// TODO test if this indexing performs worse compared to the above version
impl Index<usize> for PartdiffMatrix
{
    type Output = [f64];
    
    fn index(&self, idx: usize) -> &Self::Output
    {
        &self.matrix[idx*self.n .. (idx+1)*self.n]
    }
}

impl IndexMut<usize> for PartdiffMatrix
{
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output
    {
        &mut self.matrix[idx*self.n .. (idx+1)*self.n]
    }
}


fn usage()
{
    println!("Usage: ./partdiff [number] [method] [interlines] [func] [termination] [prec/iter]\n");
    println!("  -number:      number of threads (1 .. n)");
    println!("  -method:      calculation method (MethGaussSeidel / MethJacobi)");
    println!("  -interlines:  number of interlines (1 .. n)");
    println!("                  matrixsize = (interlines * 8) + 9");
    println!("  -func:        inference function (FuncF0 / FuncFPiSin)");
    println!("  -termination: termination condition (TermPrec, TermIter)");
    println!("                  TermPrec: sufficient precision");
    println!("                  TermIter: number of iterations");
    println!("  -prec/iter:   depending on termination:");
    println!("                  precision: 1e-4 .. 1e-20");
    println!("                  iterations: 1 .. n");
}

fn ask_params(mut args: std::env::Args) -> CalculationOptions
{
    // TODO keep authors of original c version?
    println!("============================================================");
    println!("Program for calculation of partial differential equations.  ");
    println!("============================================================");
    // println!("(c) Dr. Thomas Ludwig, TU München.");
    // println!("    Thomas A. Zochler, TU München.");
    // println!("    Andreas C. Schmidt, TU München.");
    // println!("============================================================");

    // TODO interactive arguments   
       
    args.next();

    let number: u64 = match args.next()
    {
        Some(arg) =>
        {
            let num: u64 = arg.parse().unwrap_or_else(|error|
                {
                    println!("Error: {}", error);
                    usage();
                    process::exit(1);
                });

            if num < 1
            {
                println!("Error number argument must be a positive integer");
                usage();
                process::exit(1);
            }

            num
        },
        None => 
        {
            println!("Error: incomplete arguments.");
            usage();
            process::exit(1);
        },
    };

    let method: CalculationMethod = match args.next()
    {
        Some(arg) =>
        {
            arg.parse().unwrap_or_else(|error|
                {
                    println!("Error: {}", error);
                    usage();
                    process::exit(1);
                })
        },
        None =>
        {
            println!("Error: incomplete arguments.");
            usage();
            process::exit(1);
        },
    };

    let interlines: usize = match args.next()
    {
        Some(arg) =>
        {
            arg.parse().unwrap_or_else(|error|
                {
                    println!("Error: {}", error);
                    usage();
                    process::exit(1);
                })
        },
        None =>
        {
            println!("Error: incomplete arguments.");
            usage();
            process::exit(1);
        },
    };

    let inf_func: InferenceFunction = match args.next()
    {
        Some(arg) =>
        {
            arg.parse().unwrap_or_else(|error|
                {
                    println!("Error: {}", error);
                    usage();
                    process::exit(1);
                })
        },
        None =>
        {
            println!("Error: incomplete arguments.");
            usage();
            process::exit(1);
        },
    };

    let termination: TerminationCondition = match args.next()
    {
        Some(arg) =>
        {
            arg.parse().unwrap_or_else(|error|
                {
                    println!("Error: {}", error);
                    usage();
                    process::exit(1);
                })
        },
        None =>
        {
            println!("Error: incomplete arguments.");
            usage();
            process::exit(1);
        },
    };

    match termination
    {
        TerminationCondition::TermPrec =>
        {
            match args.next()
            {
                Some(arg) =>
                {
                    let f: f64 =  arg.parse().unwrap_or_else(|error|
                        {
                            println!("Error: {}", error);
                            usage();
                            process::exit(1);
                        });

                    if (f < 1e-20) | (f > 1e-4)
                    {
                        println!("Error: termination precision must be between 1e-20 and 1e-4");
                        usage();
                        process::exit(1);
                    }

                    return CalculationOptions::new(number, method, interlines, inf_func, termination, std::u64::MAX, f);
                },
                None =>
                {
                    println!("Error: incomplete arguments.");
                    usage();
                    process::exit(1);
                },
            }
        },
        TerminationCondition::TermIter =>
        {
            match args.next()
            {
                Some(arg) =>
                {
                    let n: u64 =  arg.parse().unwrap_or_else(|error|
                        {
                            println!("Error: {}", error);
                            usage();
                            process::exit(1);
                        });

                    if n < 1
                    {
                        println!("Error: termination iterations must be > 1");
                        usage();
                        process::exit(1);
                    }

                    return CalculationOptions::new(number, method, interlines, inf_func, termination, n, 0.0);
                },
                None =>
                {
                    println!("Error: incomplete arguments.");
                    usage();
                    process::exit(1);
                },
            }
        },
    }
}

fn init_variables(options: &CalculationOptions) -> (CalculationArguments, CalculationResults)
{
    let n: usize = (options.interlines * 8) + 9 - 1;
    let num_matrices: usize = match options.method
    {
        CalculationMethod::MethGaussSeidel => 2,
        CalculationMethod::MethJacobi => 1,
    };
    let h: f64 = 1.0 as f64 / n as f64;
    let arguments = CalculationArguments::new(n, num_matrices, h);
    let results = CalculationResults::new(0,0,0.0);

    (arguments, results)
}

fn init_matrices(arguments: &mut CalculationArguments, options: &CalculationOptions)
{
    if options.inf_func == InferenceFunction::FuncF0
    {
        let matrix = &mut arguments.matrices;
        let n = arguments.n;
        let h = arguments.h;

        for g in 0 .. arguments.num_matrices as usize
        {
            for i in 0..(n+1)
            {
                matrix[g][i][0] = 1.0 - (h * i as f64);   
                matrix[g][i][n] = h * i as f64;
                matrix[g][0][i] = 1.0 - (h * i as f64);
                matrix[g][n][i] = h * i as f64;
            }
        }
    }
}

fn calculate(arguments: &mut CalculationArguments, results: &mut CalculationResults, options: &CalculationOptions)
{
    const PI: f64 = 3.141592653589793;
    const TWO_PI_SQUARE: f64 = 2.0 * PI * PI;

    let n = arguments.n;
    let h = arguments.h;

    let mut star: f64;
    let mut residuum: f64;
    let mut maxresiduum: f64;

    let mut pih: f64 = 0.0;
    let mut fpisin: f64 = 0.0;

    let mut term_iteration = options.term_iteration;

    let mut m1: usize = 0;
    let mut m2: usize = 0;

    if options.method == CalculationMethod::MethJacobi   
    {
        m1 = 0;
        m2 = 1;
    }

    if options.inf_func == InferenceFunction::FuncFPiSin
    {
        pih = PI * h;
        fpisin = 0.25 * TWO_PI_SQUARE * h * h;
    }

    while term_iteration > 0
    {
        // let matrix_in = &arguments.matrices[m1];
        // let matrix_out = &mut arguments.matrices[m2];
        
        let matrix = &mut arguments.matrices;

        maxresiduum = 0.0;

        for i in 1..n
        {
            let mut fpisin_i = 0.0;

            if options.inf_func == InferenceFunction::FuncFPiSin
            {
                fpisin_i = fpisin * (pih * i as f64).sin();
            }

            for j in 1..n
            {
                star = 0.25 * (matrix[m1][i-1][j] + matrix[m1][i+1][j] + matrix[m1][i][j-1] + matrix[m1][i][j+1]);

                if options.inf_func == InferenceFunction::FuncFPiSin
                {
                    star += fpisin_i * (pih * j as f64).sin();
                }

                if (options.termination == TerminationCondition::TermPrec) | (term_iteration == 1)
                {
                    residuum = (matrix[m1][i][j] - star).abs();
                    maxresiduum = match residuum
                    {
                        r if r < maxresiduum => maxresiduum,
                        _ => residuum,
                    };
                }

                matrix[m2][i][j] = star;
            }
        }

        results.stat_iteration += 1;
        results.stat_precision = maxresiduum;

        let tmp = m1;
        m1 = m2;
        m2 = tmp;

        match options.termination
        {
            TerminationCondition::TermPrec =>
            {
                if maxresiduum < options.term_precision
                {
                    term_iteration = 0;
                }
            },
            TerminationCondition::TermIter => term_iteration -= 1,
        }
    }

    results.m = m2;
}


fn display_statistics(arguments: &CalculationArguments, results: &CalculationResults, options: &CalculationOptions, duration: Duration)
{
    let n = arguments.n;
    
    println!("Berechnungszeit:    {:.6}", duration.as_secs_f64());
    println!("Speicherbedarf:     {:.4} MiB", ((n+1)*(n+1)*std::mem::size_of::<f64>()*arguments.num_matrices) as f64 / 1024.0 / 1024.0);
    println!("Berechnungsmethode: {:?}", options.method);
    println!("Interlines:         {}", options.interlines);
    print!("Stoerfunktion:      ");
    match options.inf_func
    {
        InferenceFunction::FuncF0 => print!("f(x,y) = 0\n"),
        InferenceFunction::FuncFPiSin => print!("f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)\n"),
    }
    print!("Terminierung:       ");
    match options.termination
    {
        TerminationCondition::TermPrec => print!("Hinreichende Genauigkeit\n"),
        TerminationCondition::TermIter => print!("Anzahl der Iterationen\n"),
    }
    println!("Anzahl Iterationen: {}", results.stat_iteration);
    println!("Norm des Fehlers:   {:.6e}", results.stat_precision);
}

fn display_matrix(arguments: &mut CalculationArguments, results: &CalculationResults, options: &CalculationOptions)
{
    let matrix = &mut arguments.matrices[results.m as usize];
    let interlines = options.interlines;

    println!("Matrix:");
    for y in 0..9 as usize
    {
        for x in 0..9 as usize
        {
            // TODO check rounding behaviour of formatting
            print!(" {:.4}", matrix[y * (interlines+1)][x * (interlines+1)]);
        }
        print!("\n");
    }
}


fn main()
{
    let options = ask_params(env::args());
    let (mut arguments, mut results) = init_variables(&options);
    init_matrices(&mut arguments, &options);
    let now = Instant::now();
    calculate(&mut arguments, &mut results, &options);
    let duration = now.elapsed();
    display_statistics(&arguments, &results, &options, duration);
    display_matrix(&mut arguments, &results, &options);
    
}
