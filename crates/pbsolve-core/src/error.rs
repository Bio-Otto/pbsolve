use thiserror::Error;

/// All errors produced by pbsolve.
#[derive(Debug, Error)]
pub enum PbError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error at line {line}: {msg}")]
    Parse { line: usize, msg: String },

    #[error("invalid input: {0}")]
    InvalidInput(String),

    #[error("backend error: {0}")]
    Backend(String),

    #[error(
        "linear solver did not converge after {iterations} iterations \
         (relative residual {residual:.3e}); increase max_iter or check the input"
    )]
    NotConverged { iterations: usize, residual: f64 },
}

pub type Result<T> = std::result::Result<T, PbError>;

pub(crate) fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(PbError::InvalidInput(msg.into()))
}
