use midas_env::env::Action;

pub const POLICY_ACTION_DIM: usize = 3;
pub const POLICY_ACTION_LABELS: [&str; POLICY_ACTION_DIM] = ["short", "flat", "long"];

pub fn policy_action_label(action_idx: i32) -> &'static str {
    match action_idx {
        0 => POLICY_ACTION_LABELS[0],
        1 => POLICY_ACTION_LABELS[1],
        _ => POLICY_ACTION_LABELS[2],
    }
}

pub fn policy_target_position(action_idx: i32) -> i32 {
    match action_idx {
        0 => -1,
        1 => 0,
        _ => 1,
    }
}

pub fn env_action_for_target(current_position: i32, target_position: i32) -> Action {
    if target_position == current_position {
        return Action::Hold;
    }

    match target_position {
        -1 => {
            if current_position > 0 {
                Action::Revert
            } else {
                Action::Sell
            }
        }
        0 => {
            if current_position > 0 {
                Action::Sell
            } else if current_position < 0 {
                Action::Buy
            } else {
                Action::Hold
            }
        }
        1 => {
            if current_position < 0 {
                Action::Revert
            } else {
                Action::Buy
            }
        }
        _ => Action::Hold,
    }
}

pub fn env_action_label(action: Action) -> &'static str {
    match action {
        Action::Buy => "buy",
        Action::Sell => "sell",
        Action::Hold => "hold",
        Action::Revert => "revert",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_policy_actions_to_expected_targets() {
        assert_eq!(policy_target_position(0), -1);
        assert_eq!(policy_target_position(1), 0);
        assert_eq!(policy_target_position(2), 1);
    }

    #[test]
    fn maps_current_position_and_target_to_env_action() {
        assert_eq!(env_action_for_target(-1, -1), Action::Hold);
        assert_eq!(env_action_for_target(-1, 0), Action::Buy);
        assert_eq!(env_action_for_target(-1, 1), Action::Revert);
        assert_eq!(env_action_for_target(0, -1), Action::Sell);
        assert_eq!(env_action_for_target(0, 0), Action::Hold);
        assert_eq!(env_action_for_target(0, 1), Action::Buy);
        assert_eq!(env_action_for_target(1, -1), Action::Revert);
        assert_eq!(env_action_for_target(1, 0), Action::Sell);
        assert_eq!(env_action_for_target(1, 1), Action::Hold);
    }
}
