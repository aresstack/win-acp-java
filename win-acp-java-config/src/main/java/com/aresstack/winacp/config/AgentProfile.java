package com.aresstack.winacp.config;

/**
 * Metadata and identity of a single agent profile.
 * <p>
 * An agent profile defines who the agent is: its name, description,
 * system role, and the set of capabilities it is allowed to use.
 */
public class AgentProfile {

    private String id;
    private String name;
    private String description;
    private String systemRole;

    public AgentProfile() {}

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getSystemRole() { return systemRole; }
    public void setSystemRole(String systemRole) { this.systemRole = systemRole; }
}

