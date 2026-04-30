use uuid::Uuid;

use super::{
    chat_actions, confirmation_flow, event, filter_events, response, Agent, ChatRequest,
    ChatResponse, Event,
};

impl Agent {
    pub async fn chat(&self, request: ChatRequest) -> ChatResponse {
        let run_id = request.run_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let max_steps = request.max_steps.unwrap_or(self.config.agent_max_steps);
        let visible = request.visible_event_kinds.clone();
        let mut events = vec![event("user", request.message.clone(), None, None)];
        let base_prior = self.transcript(&run_id).unwrap_or_default();
        if let Some((assistant, stop_reason)) = confirmation_flow::respond(
            &request.message,
            &base_prior,
            &self.config,
            &self.memory,
            &run_id,
            &mut events,
        )
        .await
        {
            self.persist(&run_id, &events);
            return response(run_id, assistant, filter_events(&events, &visible), &stop_reason);
        }
        if !self.model.is_reachable().await {
            events.push(event(
                "error",
                "model server unreachable".into(),
                None,
                None,
            ));
            self.persist(&run_id, &events);
            return response(
                run_id,
                "model unavailable".into(),
                filter_events(&events, &visible),
                "model_error",
            );
        }
        let mut prior = base_prior.clone();
        prior.extend(events.clone());
        let mut assistant = String::new();
        let mut stop_reason = "max_steps".to_string();
        let mut last_action = String::new();
        for step in 1..=max_steps {
            let action = match self.next_action(&run_id, &prior, step).await {
                Ok(action) => action,
                Err(error) => {
                    events.push(event("error", error.clone(), None, Some(step)));
                    stop_reason = chat_actions::model_stop_reason(&error);
                    break;
                }
            };
            let signature = action.signature();
            if action.tool != "agent.finish" && signature == last_action {
                events.push(event(
                    "error",
                    "repeated identical non-terminal action".into(),
                    None,
                    Some(step),
                ));
                stop_reason = "repeat_action".into();
                break;
            }
            last_action = signature;
            if let Some(reasoning) = action.reasoning.clone().filter(|v| !v.is_empty()) {
                events.push(event("reasoning", reasoning, None, Some(step)));
            }
            if self
                .handle_action(
                    action,
                    &run_id,
                    step,
                    max_steps,
                    &base_prior,
                    &mut prior,
                    &mut events,
                    &mut assistant,
                    &mut stop_reason,
                )
                .await
            {
                break;
            }
        }
        if assistant.is_empty() {
            assistant = format!("agent stopped: {stop_reason}");
        }
        self.persist(&run_id, &events);
        response(
            run_id,
            assistant,
            filter_events(&events, &visible),
            &stop_reason,
        )
    }

    async fn handle_action(
        &self,
        action: super::Action,
        run_id: &str,
        step: usize,
        max_steps: usize,
        base_prior: &[Event],
        prior: &mut Vec<Event>,
        events: &mut Vec<Event>,
        assistant: &mut String,
        stop_reason: &mut String,
    ) -> bool {
        match action.tool.as_str() {
            "agent.finish" => {
                chat_actions::finish_action(action, step, events, assistant, stop_reason)
            }
            "agent.think" => chat_actions::think_action(
                action,
                step,
                max_steps,
                base_prior,
                prior,
                events,
                stop_reason,
            ),
            "agent.request_confirmation" => {
                chat_actions::confirm_action(action, step, events, assistant, stop_reason)
            }
            _ => {
                self.tool_action(
                    action,
                    run_id,
                    step,
                    max_steps,
                    base_prior,
                    prior,
                    events,
                    stop_reason,
                )
                .await
            }
        }
    }

    async fn tool_action(
        &self,
        action: super::Action,
        run_id: &str,
        step: usize,
        max_steps: usize,
        base_prior: &[Event],
        prior: &mut Vec<Event>,
        events: &mut Vec<Event>,
        stop_reason: &mut String,
    ) -> bool {
        if super::confirmation::is_mutation(&action.tool) {
            events.push(event(
                "error",
                "resource mutation requires confirmation".into(),
                Some(action.tool),
                Some(step),
            ));
            *stop_reason = "confirmation_required".into();
            return true;
        }
        let result = self.action_tool(action, run_id, step, events).await;
        *prior = base_prior.to_vec();
        prior.extend(events.clone());
        if let Err(error) = result {
            events.push(event("error", error, None, Some(step)));
            *stop_reason = "invalid_action".into();
            return true;
        }
        if step == max_steps {
            *stop_reason = "max_steps".into();
        }
        false
    }
}
