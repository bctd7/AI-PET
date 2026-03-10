-- AI-PET 数据库：NPC 信息表 + 对话记录表
-- 数据库名：ai-pet

CREATE DATABASE IF NOT EXISTS `ai-pet` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE `ai-pet`;

-- ------------------------------------------------------------
-- NPC 基本信息表（与 modules/llm/model.NpcInfoModel 一致）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `npc` (
    `id` INT AUTO_INCREMENT PRIMARY KEY,
    `npc_id` VARCHAR(64) NOT NULL COMMENT '业务主键，唯一标识 NPC',
    `npc_name` VARCHAR(64) NOT NULL COMMENT 'NPC 名称',
    `npc_role_type` VARCHAR(32) NOT NULL DEFAULT 'default' COMMENT '角色类型，用于派发 handler 与检索集合(如 chef)',
    `npc_system_prompt` TEXT DEFAULT NULL COMMENT '系统提示词/人设描述',
    `npc_is_active` TINYINT(1) NOT NULL DEFAULT 1 COMMENT '是否启用',
    `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',
    `updated_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) ON UPDATE CURRENT_TIMESTAMP(3) COMMENT '更新时间',

    UNIQUE KEY `uk_npc_id` (`npc_id`),
    INDEX `idx_role_type` (`npc_role_type`),
    INDEX `idx_active` (`npc_is_active`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='NPC 基础信息';

-- ------------------------------------------------------------
-- 对话摘要表（本地场景：按 NPC 存最近摘要）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `chat_history` (
    `id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    `npc_id` VARCHAR(15) NOT NULL COMMENT 'NPC 业务主键',
    `summary` TEXT NOT NULL COMMENT '该轮对话摘要',
    `created_at` DATETIME(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3) COMMENT '创建时间',

    INDEX `idx_npc_created_at` (`npc_id`, `created_at`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='对话摘要记录';
