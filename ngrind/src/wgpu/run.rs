use super::vanity::VanityConfig;
use super::{Gpu, WgpuPow, WgpuVanity};
use crate::PowFound;
use crate::control::{Control, OnFound, Phase};
use crate::event::{Template, leading_zero_bits, unix_now};
use crate::pattern::Pattern;
use secp256k1::Keypair;
use std::sync::Arc;
use web_time::{Duration, Instant};

const REFRESH: Duration = Duration::from_secs(30);

pub async fn vanity(
    ctl: &dyn Control,
    pattern: &Pattern,
    adapter: usize,
    found: OnFound<'_>,
) -> Result<(), String> {
    ctl.phase(Phase::Setup);
    let gpu = Arc::new(Gpu::open(adapter).await.map_err(|e| e.to_string())?);
    ctl.log(&format!("starting on {}", gpu.info.label()));
    let mut miner =
        WgpuVanity::new(gpu, pattern, VanityConfig::default()).map_err(|e| e.to_string())?;
    ctl.phase(Phase::Search);
    let (mut counted, mut rejected) = (0u64, 0u64);
    while !ctl.stopped() {
        let hits = miner.step().await.map_err(|e| e.to_string())?;
        let now = miner.keys_generated();
        ctl.add(now - counted);
        counted = now;
        for (secret, public) in hits {
            found(secret, public);
        }
        if miner.rejected() > rejected {
            rejected = miner.rejected();
            ctl.log(&format!(
                "discarded a match that failed the check ({rejected})"
            ));
        }
    }
    Ok(())
}

pub async fn pow(
    ctl: &dyn Control,
    template: &Template,
    signer: Option<&Keypair>,
    adapter: usize,
    found: &(dyn Fn(PowFound) + Sync),
) -> Result<(), String> {
    ctl.phase(Phase::Setup);
    let gpu = Arc::new(Gpu::open(adapter).await.map_err(|e| e.to_string())?);
    let mut miner = WgpuPow::new(gpu);
    let mut created_at = unix_now();
    let split = template.split(created_at);
    miner
        .init(&split.prefix, &split.suffix, template.difficulty)
        .map_err(|e| e.to_string())?;
    ctl.log(&format!("starting on {}", miner.device_name()));
    ctl.phase(Phase::Search);

    let mut counted = 0u64;
    let mut refreshed = Instant::now();
    let mut renew = false;
    while !ctl.stopped() {
        let hit = match miner.mine().await {
            Ok(hit) => hit,
            Err(super::Error::Exhausted) => {
                renew = true;
                None
            }
            Err(e) => return Err(e.to_string()),
        };
        let total = miner.attempts();
        ctl.add(total - counted);
        counted = total;

        if let Some(hit) = hit {
            let id = template.id_of(created_at, &template.mined(hit.nonce));
            if id != hit.event_id || leading_zero_bits(&id) < template.difficulty {
                return Err("the nonce from the GPU does not reproduce the event id".into());
            }
            found(PowFound::verified(template, created_at, hit.nonce, signer)?);
            renew = true;
        }

        if renew || refreshed.elapsed() >= REFRESH {
            if ctl.stopped() {
                break;
            }
            renew = false;
            created_at = unix_now().max(created_at + 1);
            let split = template.split(created_at);
            miner
                .init(&split.prefix, &split.suffix, template.difficulty)
                .map_err(|e| e.to_string())?;
            counted = 0;
            refreshed = Instant::now();
        }
    }
    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn vanity_blocking(
    ctl: &dyn Control,
    pattern: &Pattern,
    adapter: usize,
    found: OnFound<'_>,
) -> Result<(), String> {
    pollster::block_on(vanity(ctl, pattern, adapter, found))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn pow_blocking(
    ctl: &dyn Control,
    template: &Template,
    signer: Option<&Keypair>,
    adapter: usize,
    found: &(dyn Fn(PowFound) + Sync),
) -> Result<(), String> {
    pollster::block_on(pow(ctl, template, signer, adapter, found))
}
