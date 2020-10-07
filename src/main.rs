use std::time::{Instant, Duration};
use std::ops::{Index,IndexMut};
use std::process;
use std::env;
use std::vec;

// The supported calculation Algorithms
// Gauss Seidel working on the same matrix
// Jacobi using in and out matrices
#[derive(Debug, PartialEq)]
enum CalculationMethod
{
    MethGaussSeidel,
    MethJacobi,
}

// For parsing command line arguments
impl std::str::FromStr for CalculationMethod
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err>
    {
        match s
        {
            "MethGaussSeidel" | "1" => Ok(CalculationMethod::MethGaussSeidel),
            "MethJacobi" | "2" => Ok(CalculationMethod::MethJacobi),
            _ => Err(format!("'{}' is not a valid value for CalculationMethod", s)),
        }
    }
}

// The supported inference functions used during calculation
// F0:     f(x,y) = 0
// FPiSin: f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)
#[derive(Debug, PartialEq)]
enum InferenceFunction
{
    FuncF0,
    FuncFPiSin,
}

// For parsing command line arguments
impl std::str::FromStr for InferenceFunction
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err>
    {
        match s
        {
            "FuncF0" | "1" => Ok(InferenceFunction::FuncF0),
            "FuncFPiSin" | "2" => Ok(InferenceFunction::FuncFPiSin),
            _ => Err(format!("'{}' is not a valid value for InferenceFunction", s)),
        }
    }
}


// The supported termination conditions
// TermPrec: terminate after set precision is reached
// TermIter: terminate after set amount of iterations
#[derive(Debug, PartialEq)]
enum TerminationCondition
{
    TermPrec,
    TermIter,
}

// For parsing command line arguments
impl std::str::FromStr for TerminationCondition
{
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err>
    {
        match s
        {
            "TermPrec" | "1" => Ok(TerminationCondition::TermPrec),
            "TermIter" | "2" => Ok(TerminationCondition::TermIter),
            _ => Err(format!("'{}' is not a valid value for TerminationCondition", s)),
        }
    }
}


// Data structure for storing the given parameters for the calculation
#[derive(Debug)]
struct CalculationOptions
{
    number: u64,                        // number of threads
    method: CalculationMethod,          // Gauss Seidel or Jacobi method of iteration
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


// Data structure for storing the the data needed during calculation
#[derive(Debug)]
struct CalculationArguments
{
    n: usize,                       // Number of spaces between lines (lines=n+1)
    num_matrices: usize,            // number of matrices
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


// Data structure for storing result data of the calculation
#[derive(Debug)]
struct CalculationResults
{
    m: usize,             // Index of matrix that holds the final state
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


// Simple data structure for a 2D matrix
// Has an efficient continuous 1D memory layout
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

// Implementation of Index and IndexMut traits for the matrix
// 2d-array-indexing allows access to matrix elements with following syntax:
//   matrix[[x,y]]
//
// This version is used if the crate is build with: --features "2d-array-indexing"
// 
// Also supports switching between indexing with or without bounds checking
// This can be set by building the crate with or without: --features "unsafe-indexing"
#[cfg(feature = "2d-array-indexing")]
impl Index<[usize; 2]> for PartdiffMatrix
{
    type Output = f64;

    fn index(&self, idx: [usize; 2]) -> &Self::Output
    {       
        #[cfg(not(feature = "unsafe-indexing"))]
        {
            &self.matrix[idx[0] * self.n + idx[1]]
        }
        #[cfg(feature = "unsafe-indexing")]
        unsafe
        {
            &self.matrix.get_unchecked(idx[0] * self.n + idx[1])
        }
    }
}

#[cfg(feature = "2d-array-indexing")]
impl IndexMut<[usize; 2]> for PartdiffMatrix
{
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut Self::Output
    {
        #[cfg(not(feature = "unsafe-indexing"))]
        {
            &mut self.matrix[idx[0] * self.n + idx[1]]
        }
        #[cfg(feature = "unsafe-indexing")]
        unsafe
        {
            self.matrix.get_unchecked_mut(idx[0] * self.n + idx[1])
        }
    }
}

// Implementation of Index and IndexMut traits for the matrix
// C-style-indexing allows access to matrix elements with following syntax:
//   matrix[x][y]
//
// This version is used if the crate is build with: --features "C-style-indexing"
// 
// Also supports switching between indexing with or without bounds checking
// This can be set by building the crate with or without: --features "unsafe-indexing"
#[cfg(feature = "C-style-indexing")]
impl Index<usize> for PartdiffMatrix
{
    type Output = [f64];

    fn index(&self, idx: usize) -> &Self::Output
    {
        #[cfg(not(feature = "unsafe-indexing"))]
        {
            &self.matrix[idx*self.n .. (idx+1)*self.n]
        }
        #[cfg(feature = "unsafe-indexing")]
        unsafe
        {
            &self.matrix.get_unchecked(idx*self.n .. (idx+1)*self.n)
        }
    }
}

#[cfg(feature = "C-style-indexing")]
impl IndexMut<usize> for PartdiffMatrix
{
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output
    {
        #[cfg(not(feature = "unsafe-indexing"))]
        {
            &mut self.matrix[idx*self.n .. (idx+1)*self.n]
        }
        #[cfg(feature = "unsafe-indexing")]
        unsafe
        {
            self.matrix.get_unchecked_mut(idx*self.n .. (idx+1)*self.n)
        }
    }
}


// Display help message to show the required command line arguments to run the binary
fn usage()
{
    println!("Usage: ./rust_partdiff [number] [method] [interlines] [func] [termination] [prec/iter]\n");
    println!("  -number:      number of threads (1 .. n)");
    println!("  -method:      calculation method (MethGaussSeidel/MethJacobi OR 1/2)");
    println!("  -interlines:  number of interlines (1 .. n)");
    println!("                  matrixsize = (interlines * 8) + 9");
    println!("  -func:        inference function (FuncF0/FuncFPiSin OR 1/2)");
    println!("  -termination: termination condition (TermPrec/TermIter OR 1/2)");
    println!("                  TermPrec: sufficient precision");
    println!("                  TermIter: number of iterations");
    println!("  -prec/iter:   depending on termination:");
    println!("                  precision: 1e-4 .. 1e-20");
    println!("                  iterations: 1 .. n");
}


// Helper function to parse command line arguments
fn parse_arg<U>(arg: Option<String>) -> U
where U: std::str::FromStr,
      <U as std::str::FromStr>::Err: std::fmt::Display
{
    let ret: U = match arg
    {
        Some(a) =>
        {
            a.parse().unwrap_or_else(|error|
                {
                    eprintln!("Error: {}", error);
                    usage();
                    process::exit(1);
                })
        },
        None =>
        {
            eprintln!("Error: incomplete arguments.");
            usage();
            process::exit(1);
        },
    };
    ret
}

// Parsing of command line arguments
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
    
    let number: u64 = parse_arg(args.next());
    if number < 1
    {
        eprintln!("Error number argument must be a positive integer");
        usage();
        process::exit(1);
    }

    let method: CalculationMethod = parse_arg(args.next());

    let interlines: usize = parse_arg(args.next());

    let inf_func: InferenceFunction = parse_arg(args.next());

    let termination: TerminationCondition = parse_arg(args.next());

    // Check for the meaning of the last argument
    match termination
    {
        TerminationCondition::TermPrec =>
        {
            let prec: f64 = parse_arg(args.next());
            if (prec < 1e-20) | (prec > 1e-4)
            {
                eprintln!("Error: termination precision must be between 1e-20 and 1e-4");
                usage();
                process::exit(1);
            }
            return CalculationOptions::new(number, method, interlines, inf_func, termination, std::u64::MAX, prec);
        },
        TerminationCondition::TermIter =>
        {
            let iterations = parse_arg(args.next());
            if iterations < 1
            {
                eprintln!("Error: termination iterations must be > 0");
                usage();
                process::exit(1);
            }
            return CalculationOptions::new(number, method, interlines, inf_func, termination, iterations, 0.0);
        },
    }
}


// Determine calculation arguments and initialize calculation results
fn init_variables(options: &CalculationOptions) -> (CalculationArguments, CalculationResults)
{
    let n: usize = (options.interlines * 8) + 9 - 1;
    let num_matrices: usize = match options.method
    {
        CalculationMethod::MethGaussSeidel => 1,
        CalculationMethod::MethJacobi => 2,
    };
    let h: f64 = 1.0 as f64 / n as f64;
    let arguments = CalculationArguments::new(n, num_matrices, h);
    let results = CalculationResults::new(0,0,0.0);

    (arguments, results)
}


// Initialize the matrix borders according to the used inference function
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
                #[cfg(feature = "2d-array-indexing")]
                {
                    matrix[g][[i,0]] = 1.0 - (h * i as f64);   
                    matrix[g][[i,n]] = h * i as f64;
                    matrix[g][[0,i]] = 1.0 - (h * i as f64);
                    matrix[g][[n,i]] = h * i as f64;
                }
                #[cfg(feature = "C-style-indexing")]
                {
                    matrix[g][i][0] = 1.0 - (h * i as f64);   
                    matrix[g][i][n] = h * i as f64;
                    matrix[g][0][i] = 1.0 - (h * i as f64);
                    matrix[g][n][i] = h * i as f64;
                }
            }
        }
    }
}


// Main calculation
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

    // for distinguishing between old and new state of the matrix if two matrices are used
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
                #[cfg(feature = "2d-array-indexing")]
                {
                    star = 0.25 * (matrix[m1][[i-1,j]] + matrix[m1][[i+1,j]] +
                        matrix[m1][[i,j-1]] + matrix[m1][[i,j+1]]);
                }
                #[cfg(feature = "C-style-indexing")]
                {
                    star = 0.25 * (matrix[m1][i-1][j] + matrix[m1][i+1][j] +
                        matrix[m1][i][j-1] + matrix[m1][i][j+1]);
                }

                if options.inf_func == InferenceFunction::FuncFPiSin
                {
                    star += fpisin_i * (pih * j as f64).sin();
                }

                if (options.termination == TerminationCondition::TermPrec) | (term_iteration == 1)
                {
                    #[cfg(feature = "2d-array-indexing")]
                    {
                        residuum = (matrix[m1][[i,j]] - star).abs();
                    }
                    #[cfg(feature = "C-style-indexing")]
                    {
                        residuum = (matrix[m1][i][j] - star).abs();
                    }

                    maxresiduum = match residuum
                    {
                        r if r < maxresiduum => maxresiduum,
                        _ => residuum,
                    };
                }

                #[cfg(feature = "2d-array-indexing")]
                {
                    matrix[m2][[i,j]] = star;
                }
                #[cfg(feature = "C-style-indexing")]
                {
                    matrix[m2][i][j] = star;
                }
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


// Display important information about the calculation
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


// Beschreibung der Funktion displayMatrix:                              
//                                                                       
// Die Funktion displayMatrix gibt eine Matrix                           
// in einer "ubersichtlichen Art und Weise auf die Standardausgabe aus.  
//                                                                       
// Die "Ubersichtlichkeit wird erreicht, indem nur ein Teil der Matrix   
// ausgegeben wird. Aus der Matrix werden die Randzeilen/-spalten sowie  
// sieben Zwischenzeilen ausgegeben.                                     
fn display_matrix(arguments: &mut CalculationArguments, results: &CalculationResults, options: &CalculationOptions)
{
    let matrix = &mut arguments.matrices[results.m as usize];
    let interlines = options.interlines;

    println!("Matrix:");
    for y in 0..9 as usize
    {
        for x in 0..9 as usize
        {
            #[cfg(feature = "2d-array-indexing")]
            {
                print!(" {:.4}", matrix[[y * (interlines+1),x * (interlines+1)]]);
            }
            #[cfg(feature = "C-style-indexing")]
            {
                print!(" {:.4}", matrix[y * (interlines+1)][x * (interlines+1)]);
            }
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
