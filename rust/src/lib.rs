//! Elle's local-first foundation, selectively adapted from the owner's project.
//!
//! This crate does not yet expose an Entra-authenticated HTTP listener.

pub mod archive;
pub mod auth;
pub mod azure;
pub mod cognitive;
pub mod embeddings;
pub mod encryption;
pub mod error;
pub mod file_repository;
pub mod identity;
pub mod mcp;
pub mod memory;
pub mod personality;
pub mod repository;
pub mod server;
pub mod service;
pub mod telemetry;
pub mod wisdom;
