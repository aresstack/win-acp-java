package com.aresstack.winacp.config;

/**
 * Policy governing when human approval is required.
 */
public class ApprovalPolicy {

    private boolean enabled = true;
    private boolean approvalRequiredByDefault = false;

    public ApprovalPolicy() {}

    public boolean isEnabled() { return enabled; }
    public void setEnabled(boolean enabled) { this.enabled = enabled; }

    public boolean isApprovalRequiredByDefault() { return approvalRequiredByDefault; }
    public void setApprovalRequiredByDefault(boolean approvalRequiredByDefault) {
        this.approvalRequiredByDefault = approvalRequiredByDefault;
    }
}

